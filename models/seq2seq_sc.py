from typing import Optional, Tuple, Union
from transformers import TFBartPretrainedModel, TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartMainLayer, BartConfig, shift_tokens_right, TFBartEncoder
from transformers.modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqModelOutput
from transformers.modeling_tf_utils import unpack_inputs, TFModelInputType, DUMMY_INPUTS
import tensorflow as tf

import sionna
sionna.Config.xla_compat=True

from sionna.channel import AWGN, FlatFadingChannel
from sionna.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.mimo import mf_equalizer
from sionna.utils import ebnodb2no, expand_to_rank
from .utils import tensor_to_binary, binary_to_tensor
import numpy as np
from transformers.utils import logging

class TFSeq2SeqSCEncoderChannel(tf.keras.layers.Layer):

    def __init__(self, 
            encoder: TFBartEncoder, 
            ebno_db,
            polar_k=512,
            polar_n=1024,
            polar_decoder_type='SC',
            polar_decoder_list_size=8,
            num_bits_per_symbol=4,
            channel_type = 'AWGN',
            channel_num_tx_ant=1, 
            channel_num_rx_ant=1):
        # NOTE: setting layer name as follows seems strange, 
        # but it allows HuggingFace to load pretrained weight properly
        super().__init__(name='model/model/')
        logger = logging.get_logger("transformers")
        self.config = encoder.config
        self.bart_encoder = encoder
        
        # channel encoder/decoder, channel noise
        assert ebno_db is not None
        self.ebno_db = float(ebno_db)
        self.k = polar_k
        self.n = polar_n
        logger.info(f'{self.ebno_db=}')
        logger.info(f'{self.k=}')
        logger.info(f'{self.n=}')
        self.channel_encoder = Polar5GEncoder(k=self.k, n=self.n)

        constellation = Constellation("qam",
                                    num_bits_per_symbol,
                                    trainable=False)
        logger.info(f'Constellation: type={constellation._constellation_type} {num_bits_per_symbol=} trainable={constellation._trainable}')
        self.num_bits_per_symbol = num_bits_per_symbol
        self.mapper = Mapper(constellation=constellation)
        if channel_type == 'AWGN':
            self.channel = AWGN()
            self.channel_num_tx_ant = 1
            self.channel_num_rx_ant = 1
        elif channel_type == 'FlatFadingChannel':
            self.channel = FlatFadingChannel(channel_num_tx_ant, channel_num_rx_ant, add_awgn=True, return_channel=True)
            self.channel_num_tx_ant = channel_num_tx_ant
            self.channel_num_rx_ant = channel_num_rx_ant
        else:
            raise ValueError(f"Invalid channel type: {channel_type}")
        logger.info(f'{channel_type=}')
        self.demapper = Demapper("app", constellation=constellation)
        self.channel_decoder = Polar5GDecoder(
            self.channel_encoder,
            dec_type=polar_decoder_type,
            list_size=polar_decoder_list_size)
        self.coderate = self.k / self.n
        logger.info(f'{self.coderate=}')

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        encoder_outputs = self.bart_encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # add channel noise
        encoder_outputs.last_hidden_state = \
            self._add_channel_noise(encoder_outputs.last_hidden_state)
        
        # denoise tensor
        encoder_outputs.last_hidden_state = \
            tf.math.tanh(encoder_outputs.last_hidden_state)
        tf.debugging.assert_all_finite(
            encoder_outputs.last_hidden_state, 
            'should not have nan/inf/-inf after tanh')
        return encoder_outputs

    @tf.function
    def _add_channel_noise(self, input):
        encoder_output_shape = tf.shape(input)

        # Channel encoder
        encoder_output_binary = tensor_to_binary(input)
        encoder_output_binary = tf.reshape(encoder_output_binary, (-1, self.k))
        codewords = self.channel_encoder(encoder_output_binary)

        # Modulation
        x = self.mapper(codewords)

        #####################
        # Channel
        #####################
        no = ebnodb2no(self.ebno_db, self.num_bits_per_symbol, self.coderate)
        no = expand_to_rank(no, 2)
        if isinstance(self.channel, FlatFadingChannel):
            shape = tf.shape(x)
            x = tf.reshape(x, (-1, self.channel_num_tx_ant))
            y, h = self.channel([x, no])
            s = tf.complex(no*tf.eye(self.channel_num_rx_ant, self.channel_num_rx_ant), 0.0)

            x_hat, no_eff = mf_equalizer(y, h, s)

            x_hat = tf.reshape(x_hat, shape)
            no_eff = tf.reshape(no_eff, shape)

            y = x_hat
            no = no_eff
        else:
            y = self.channel([x, no])

        #####################
        # Receiver
        #####################
        # Demodulation
        llr = self.demapper([y, no])
        llr = tf.reshape(llr, (-1, self.n))

        # Channel decoder
        received_codewords = self.channel_decoder(llr)

        received_encoder_output = binary_to_tensor(received_codewords)
        received_encoder_output = tf.reshape(received_encoder_output,
                                             encoder_output_shape)
        return received_encoder_output

class TFSeq2SeqSCMainLayer(tf.keras.layers.Layer):

    def __init__(self,
                 config: BartConfig,
                 bart_main_layer: TFBartMainLayer,
                 ebno_db,
                 polar_k=512,
                 polar_n=1024,
                 polar_decoder_type='SC',
                 polar_decoder_list_size=8,
                 num_bits_per_symbol=4,
                 channel_type = 'AWGN',
                 channel_num_tx_ant=1, 
                 channel_num_rx_ant=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.shared = bart_main_layer.get_input_embeddings()

        # semantic encoders
        self.encoder = TFSeq2SeqSCEncoderChannel(
            encoder=bart_main_layer.encoder,
            ebno_db=ebno_db,
            polar_k=polar_k,
            polar_n=polar_n,
            polar_decoder_type=polar_decoder_type,
            polar_decoder_list_size=polar_decoder_list_size,
            num_bits_per_symbol=num_bits_per_symbol,
            channel_type=channel_type,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant)
        self.decoder = bart_main_layer.decoder

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    

    @unpack_inputs
    def call(self,
             input_ids: Optional[TFModelInputType] = None,
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_attention_mask: Optional[Union[np.ndarray,
                                                    tf.Tensor]] = None,
             decoder_position_ids: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             cross_attn_head_mask: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
             past_key_values: Optional[Tuple[Tuple[Union[np.ndarray,
                                                         tf.Tensor]]]] = None,
             inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_inputs_embeds: Optional[Union[np.ndarray,
                                                   tf.Tensor]] = None,
             use_cache: Optional[bool] = None,
             output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None,
             return_dict: Optional[bool] = None,
             training: Optional[bool] = False,
             **kwargs) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id,
                self.config.decoder_start_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs,
                                            TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None,
            )

        # If the user passed a TFBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TFSeq2SeqSCModel(TFBartPretrainedModel):

    def __init__(self,
                 config: BartConfig,
                 load_weight_prefix=None,
                 ebno_db=None,
                 polar_k=512,
                 polar_n=1024,
                 polar_decoder_type='SC',
                 polar_decoder_list_size=8,
                 num_bits_per_symbol=4,
                 channel_type = 'AWGN',
                 channel_num_tx_ant = 1,
                 channel_num_rx_ant = 1, 
                 *inputs,
                 **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bart_layer = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        # self.bart_layer(DUMMY_INPUTS)
        self.model = TFSeq2SeqSCMainLayer(
            config,
            ebno_db=ebno_db,
            bart_main_layer=self.bart_layer,
            polar_k=polar_k,
            polar_n=polar_n,
            polar_decoder_type=polar_decoder_type,
            polar_decoder_list_size=polar_decoder_list_size,
            num_bits_per_symbol=num_bits_per_symbol,
            channel_type=channel_type,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant
            )

    @unpack_inputs
    def call(self,
             input_ids: Optional[TFModelInputType] = None,
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_attention_mask: Optional[Union[np.ndarray,
                                                    tf.Tensor]] = None,
             decoder_position_ids: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             cross_attn_head_mask: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
             past_key_values: Optional[Tuple[Tuple[Union[np.ndarray,
                                                         tf.Tensor]]]] = None,
             inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_inputs_embeds: Optional[Union[np.ndarray,
                                                   tf.Tensor]] = None,
             use_cache: Optional[bool] = None,
             output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None,
             return_dict: Optional[bool] = None,
             training: Optional[bool] = False,
             **kwargs) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs


class TFSeq2SeqSCForConditionalGeneration(TFBartForConditionalGeneration):
    def __init__(self,
                 config,
                 ebno_db=None,
                 polar_k=512,
                 polar_n=1024,
                 polar_decoder_type='SC',
                 polar_decoder_list_size=8,
                 num_bits_per_symbol=4,
                 channel_type = 'AWGN',
                 channel_num_tx_ant = 1,
                 channel_num_rx_ant = 1, 
                 *inputs,
                 **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFSeq2SeqSCMainLayer(
            config,
            bart_main_layer=self.model,
            ebno_db=ebno_db,
            polar_k=polar_k,
            polar_n=polar_n,
            polar_decoder_type=polar_decoder_type,
            polar_decoder_list_size=polar_decoder_list_size,
            num_bits_per_symbol=num_bits_per_symbol,
            channel_type=channel_type,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant,
            name="model")

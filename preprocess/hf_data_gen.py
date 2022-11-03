# exampel file format
# --------------------
# text,summary
# "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
# "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
# "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
import csv

class HFDataGenerator:

    FIELDNAMES = ['text', 'summary']

    def __init__(self):
        self.rows = []

    def add(self, text, summary):
        if isinstance(text, str) and isinstance(summary, str):
            self.rows.append({
                'text': f'{text}',
                'summary': f'{summary}',
            })
        elif isinstance(text, list) and isinstance(summary, list):
            assert len(text) == len(summary), "Different text and summary list"
            rows = map(lambda ts: {'text': ts[0], 'summary': ts[1]}, zip(text, summary))
            self.rows.extend(rows)
        else:
            assert False, "No such case"
        

    def dump(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(self.rows) 

def test():
    data_path = "/tmp/hf_data_test.csv"
    gen = HFDataGenerator()
    gen.add("text1", "summary1")
    gen.add("text2", "summary2")
    gen.dump(data_path)
    
    expected = """"text","summary"
"text1","summary1"
"text2","summary2"
"""
    with open( data_path) as f:
        actual = f.read()
    assert actual == expected, f"""'{actual}'"""

if __name__ == '__main__':
    test()
    

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    site = """
<font face="Arial, sans-serif">
<a href = http://github.com/nervanasystems/neon >neon</a> (Github)<br><br>
<a href = http://github.com/nervanasystems/meetup >iPython notebooks</a> (Github)<br><br>
<a href = https://s3-us-west-1.amazonaws.com/nervana-meetup/labeledTrainData.tsv >IMDB Dataset</a> (right click and download)<br>
</font>
"""
    return site

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    numbers = list(range(1, 11))
    table_rows = ''.join([f'<tr><td>{i}</td></tr>' for i in numbers])
    table = f'<table>{table_rows}</table>'
    return table

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=54501)

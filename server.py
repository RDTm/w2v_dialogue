from flask import Flask, render_template, request, session, escape
import dialogue_manager

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
	query = request.form['name']
	resp = DM.respond(query)
        session['list'].append(query)
	session['list'].append(resp)
        return render_template('form.html', list=session['list'])
    else:
        session['list'] = []
        return render_template('form.html')

if __name__ == '__main__':
    # set the secret key.  keep this really secret:
    DM = dialogue_manager.dialogue()
    app.secret_key = 'mytest'
    app.run(host='0.0.0.0', debug=True)

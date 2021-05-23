## This is the Code For Research Paper Recommendation Software's main file; This file contains the code for 
## FLask application used for the Softwares web application.
import subprocess
#subprocess.run('pip install -r requirements.txt',shell=False,stdout=subprocess.DEVNULL,)

from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cgi, csv
import pandas as pd
from tensorflow import keras
import mysql.connector
database = 'rsproject'

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
query = ""
results = []
userid = ""

class User(db.Model):
	id = db.Column(db.Integer,primary_key = True)
	UserID = db.Column(db.String(20),unique=True,nullable=False)
	email = db.Column(db.String(40),unique=True,nullable=False)
	password = db.Column(db.String(20),nullable=False)
	seen = db.relationship('Papers',lazy=True,backref='author')
	def __repr__(self):
		return '<User %r>' % self.UserID

class Papers(db.Model):
	id = db.Column(db.Integer,primary_key = True)
	title = db.Column(db.String(200),unique=False,nullable=False)
	UserID = db.Column(db.Integer, db.ForeignKey('user.id'),nullable=False)
	KDM = db.Column(db.Float, default=0)
	CAOT = db.Column(db.Float, default=0)
	SQM = db.Column(db.Float, default=0)
	SCA = db.Column(db.Float, default=0)
	topic = db.Column(db.Integer,default=1)
	time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
	def __repr__(self):
		return '<Paper %r>' % self.title
db.create_all()


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('rs.html', )

@app.route('/register', methods=['POST', 'GET'])
def register():
	
	print(request.form)
	id = request.form["UserID"]
	email = request.form['Email']
	password = request.form['Password']
	db = mysql.connector.connect(
	  host="localhost",
	  user="root",
	  password="",
	  database=database
	)
	cursor = db.cursor()
	sql = ("SELECT * FROM users WHERE UserID='"+str(id)+"'")
	cursor.execute(sql)
	res = cursor.fetchall()
	if(len(res)!=0):
		return "Username Already Used"
	insert = ("INSERT INTO `users` (`id`, `UserID`, `email`, `password`) VALUES (NULL, '"+id+"', '"+email+"', '"+password+"');")
	cursor.execute(insert)
	db.commit()
	cursor.close()

	return render_template('rs.html', )


@app.route('/login' , methods=['POST', 'GET'])   
def login():
	id = request.form['UserID']
	password = request.form['Password']
	db = mysql.connector.connect(
	  host="localhost",
	  user="root",
	  password="",
	  database=database
	)
	cursor = db.cursor()
	sql = ("SELECT * FROM users WHERE UserID='"+str(id)+"'")
	cursor.execute(sql)
	res = cursor.fetchall()
	cursor.close()

	if(len(res)==0):
		return "Invalide User ID"
	if(res[0][3]!=password):
		return "Incorrect Password"
	global userid 
	userid = id
	return render_template('homepage.html')


@app.route('/search' , methods=['POST', 'GET'])
def search():
	from utils import mcda,lstm,rankaggr_brute
	global query,userid
	global results
	query = request.form['search']

	import subprocess
	subprocess.run('cd main && python web_crawl.py '+query.replace(" ", "")+' && cd..',shell=True)
	
	

	################ORIGINAL#############
	db = mysql.connector.connect(
	  host="localhost",
	  user="root",
	  password="",
	  database=database
	)
	cursor = db.cursor()
	sql = ("SELECT * FROM papers WHERE UserID='"+str(userid)+"'")
	cursor.execute(sql)
	papers = cursor.fetchall()
	cursor.close()

	df = pd.read_csv('main/data.csv')
	if len(papers)<15:
		rank = mcda(df)
		results.clear()
		results = []
		for i in range(min(len(rank),10)):
			result = [df['Source title'][rank[i]], df['Abstract'][rank[i]], df['Link'][rank[i]],rank[i]]
			results.append(result)
		return render_template('results.html', results=results,query=query)
	else:
		print('LSTM\nLSTM\nLSTM')
		topic = lstm(papers)
		rank = mcda(df)
		results.clear()
		for i in range(len(rank)):
			if df['Topic'][rank[i]]==topic[0] or df['Topic'][rank[i]]==topic[1]:
				result = [df['Source title'][rank[i]], df['Abstract'][rank[i]], df['Link'][rank[i]],rank[i]]
				results.append(result)
		for i in range(len(rank)):
			if df['Topic'][rank[i]]!=topic[0] and df['Topic'][rank[i]]!=topic[1]:
				result = [df['Source title'][rank[i]], df['Abstract'][rank[i]], df['Link'][rank[i]],rank[i]]
				results.append(result)
		return render_template('results.html', results=results,query=query)




@app.route('/read/<string:num>' , methods=['POST', 'GET'])
def read(num):
	import webbrowser
	import pandas as pd
	df = pd.read_csv('main/data.csv')
	i = int(num)
	webbrowser.open_new_tab(df['Link'][i])
	title =df['Source title'][i] 
	UserID=userid
	KDM=df['KDM'][i] 
	CAOT=df['CAOT'][i]
	SQM=df['SQM'][i]
	SCA=df['SCA'][i]
	topic=df['Topic'][i]

	db = mysql.connector.connect(
	  host="localhost",
	  user="root",
	  password="",
	  database=database
	)
	cursor = db.cursor()
	sql = ("INSERT INTO `papers` (`id`, `title`, `UserID`, `KDM`, `CAOT`, `SQM`, `SCA`, `topic`, `time`) VALUES (NULL, '"+title+"','"+UserID+"', '"+str(KDM)+"','"+str(CAOT)+"','"+str(SQM)+"','"+str(SCA)+"','"+str(topic)+"', current_timestamp());")
	
	
	try:
		cursor.execute(sql)
		db.commit()
		print("Paper added")
	except:
		print("Error")
	cursor.close()
	return render_template('results.html', results=results,query=query)

@app.route('/back' , methods=['POST', 'GET'])
def back():
	return render_template('homepage.html')



if __name__ == "__main__":
	from waitress import serve
	serve(app, host="0.0.0.0", port=8080)
	#app.run(debug=True)



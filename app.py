from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cgi, csv
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
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
	user = User.query.filter_by(UserID='aitra').first()
	if(user!=None):
		return "Username Already Used"
	new_user = User(UserID=id,email=email,password=password)
	db.session.add(new_user)
	db.session.commit()
	#csv_file = csv.reader(open('User_Details.csv', "r"), delimiter=",")
	'''
	for row in csv_file:
		if id == row[0]:
			print("Username already taken")
			return render_template('rs.html', )
	with open('User_Details.csv','a') as fd:
		fd.write(str(id)+','+str(email)+','+str(password)+'\n')'''

	return render_template('rs.html', )


@app.route('/login' , methods=['POST', 'GET'])   
def login():
	id = request.form['UserID']
	password = request.form['Password']
	user = User.query.filter_by(UserID=id).first()
	if(user==None):
		return "Invalide User ID"
	if(user.password != password):
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

	#print(query)
	import subprocess
	#subprocess.run('cd main')
	subprocess.run('cd main && python web_crawl.py '+query.replace(" ", "")+' && cd..',shell=True)
	#subprocess.run('cd..')
	
	papers = Papers.query.filter_by(UserID=userid).all()
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
		rank = rankaggr_brute(df)
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
		'''
		df1 = df[(df['Topic']==topic[0]) | (df['Topic']==topic[1])]
		df2 = df[(df['Topic']!=topic[0]) & (df['Topic']!=topic[1])]
		rank1 = mcda(df1)
		print('Len Rank1 = ',len(rank1))
		results.clear()
		results = []
		for i in range(min(len(rank1),10)):
			result = [df1['Source title'][rank1[i]], df1['Abstract'][rank1[i]], df1['Link'][rank1[i]],rank1[i]]
			results.append(result)
		if len(results)>=10: return render_template('results.html', results=results,query=query)
		rank2 = mcda(df2)
		print('Len Rank2 = ',len(rank2))
		for i in range(10-len(rank1)):
			result = [df2['Source title'][rank2[i]], df2['Abstract'][rank2[i]], df2['Link'][rank2[i]],rank2[i]]
			results.append(result)
		return render_template('results.html', results=results,query=query)'''

	'''
	csv_file = csv.reader(open('data.csv', "r"), delimiter=",")
	next(csv_file)
	for row in csv_file:
		result = [row[2],row[4],row[1]]
		results.append(result)'''

@app.route('/read/<string:num>' , methods=['POST', 'GET'])
def read(num):
	import webbrowser
	import pandas as pd
	df = pd.read_csv('main/data.csv')
	i = int(num)
	webbrowser.open_new_tab(df['Link'][i])
	new_paper = Papers(title =df['Source title'][i] ,UserID=userid,KDM=df['KDM'][i], CAOT=df['CAOT'][i],
		SQM=df['SQM'][i],SCA=df['SCA'][i],topic=df['Topic'][i])
	
	try:
		db.session.add(new_paper)
		db.session.commit()
		print("Paper added")
	except:
		print("Error")
	return render_template('results.html', results=results,query=query)

@app.route('/back' , methods=['POST', 'GET'])
def back():
	return render_template('homepage.html')



if __name__ == "__main__":
    app.run(debug=True)



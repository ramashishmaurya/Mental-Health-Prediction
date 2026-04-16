from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def getvalue():
    return {'message':'the method is calling '}
message = {'name':'ashish' , 'class':10}

@app.post('/postvalue')
def postv():
   return message 

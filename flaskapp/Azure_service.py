import urllib2
# If you are using Python 3+, import urllib instead of urllib2
import json

def getPrediction(crim, zn, lstat, age, tax, rad, black, chas, rm, nox, indus, ptratio, dis):
    data =  {
            "Inputs": {
                    "input1":
                    {
                        "ColumnNames": ["crim", "zn", "lstat", "age", "tax", "rad", "black",
                                        "chas", "nox", "rm", "indus", "ptratio", "dis"],
                        "Values": [
                                   [crim, zn, lstat, age, tax, rad, black, chas, rm, nox, indus, ptratio, dis]
                                ]
                    },        },
                "GlobalParameters": {
            }
        }

    body = str.encode(json.dumps(data))
    url = "https://ussouthcentral.services.azureml.net/workspaces/0d32a72e10464c6d9d1d1929a701f9ae/services/5b40606a931f4d39a9fa1a06d0fc02e3/execute?api-version=2.0"
    api_key = "3Ncy08XPKLc4LEH0iTAj5exjTUqtQyyZJ9tb3gfZTy/uRSM/NPb/DD+U7JkGxYy6T9iDXBd8pTsJQypo6TmrLg=="
    # Replace this with the API key for the web service

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)

        # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
        # req = urllib.request.Request(url, body, headers)
        # response = urllib.request.urlopen(req)

        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][0]
        generateResponse(prediction)
        return prediction
    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp,
        # which are useful for debugging the failure
        print(error.info())

        print(json.loads(error.read()))

def generateResponse(prediction):
    with open('/home/ubuntu/flaskapp/prediction.html','w') as myFile:
        myFile.write('<html>')
        myFile.write('<body>')
        myFile.write('<head>')
        myFile.write('The prediction result is:' + prediction)
        myFile.write('</head></br>')
        myFile.write('</br>')

        myFile.write('</body>')
        myFile.write('</html>')


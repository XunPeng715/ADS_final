import urllib2
# If you are using Python 3+, import urllib instead of urllib2
import json

def getLinearRegression(orig_loantovalue,
                  credit_score,
                  orig_upb,
                  no_borrower,
                  orig_debttoincome,
                  no_unit,
                  orig_combined_loantovalue,
                  first_time_homebuyer_flag_Y,
                  occupancy_status_I, occupancy_status_O, occupancy_status_S,
                  channel_B, channel_C, channel_R, channel_T,
                  prepayment_penalty_mortgage_flag_Y,
                  loan_purpose_C, loan_purpose_N, loan_purpose_P, ):
    data = {

        "Inputs": {

            "input1":
                {
                    "ColumnNames": ["channel_T", "no_unit", "no_borrower", "orig_loantovalue", "channel_R",
                                    "credit_score", "loan_purpose_P", "occupancy_status_S", "orig_upb",
                                    "first_time_homebuyer_flag_Y", "occupancy_status_O", "loan_purpose_N",
                                    "occupancy_status_I", "loan_purpose_C", "orig_debttoincome",
                                    "prepayment_penalty_mortgage_flag_Y"],
                    "Values": [
                        [channel_T, no_unit, no_borrower, orig_loantovalue, channel_R,
                                    credit_score, loan_purpose_P, occupancy_status_S, orig_upb,
                                    first_time_homebuyer_flag_Y, occupancy_status_O, loan_purpose_N,
                                    occupancy_status_I, loan_purpose_C, orig_debttoincome,
                                    prepayment_penalty_mortgage_flag_Y]]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/0d32a72e10464c6d9d1d1929a701f9ae/services/811c9dc4e9c040b48b4013b9a753e809/execute?api-version=2.0&details=true'
    api_key = 'FQh+J1xpVlS3SlpjdP/rR2E8Co7Tm/MobXG8hT+4U0iSReyjFR6fXEYvvt3ytqdEGanju1KH59V2P+TDxWd1Xw=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][0]
        generateResponse(prediction)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))

def getRFregression(credit_score,
                    first_time_homebuyer_flag,
                    no_unit,
                    occupancy_status,
                    orig_debttoincome,
                    orig_upb,
                    orig_loantovalue,
                    orig_interest_rate,
                    channel,
                    prepayment_penalty_mortgage_flag,
                    loan_purpose,
                    no_borrower):
    data = {
        "Inputs": {
            "input1":
                {
                    "ColumnNames": ["credit_score", "first_time_homebuyer_flag", "no_unit", "occupancy_status",
                                    "orig_debttoincome", "orig_upb", "orig_loantovalue", "orig_interest_rate",
                                    "channel", "prepayment_penalty_mortgage_flag", "loan_purpose", "no_borrower"],
                    "Values": [[credit_score,
                                first_time_homebuyer_flag,
                                no_unit,
                                occupancy_status,
                                orig_debttoincome,
                                orig_upb,
                                orig_loantovalue,
                                orig_interest_rate,
                                channel,
                                prepayment_penalty_mortgage_flag,
                                loan_purpose,
                                no_borrower] ]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))
    url = 'https://ussouthcentral.services.azureml.net/workspaces/19e3e5bdfa6b42d381df6c8b26ce71f6/services/f1eb4c6847d949b79178295b152003b9/execute?api-version=2.0&details=true'
    api_key = 'h6Q9obInvZfdSb2G/39R/Ls8qOPSCjdBVAZVTO4zNKXellWzdZkFq6yWOF2Z+A8b4Hdz0gP0B9pg0qn6XOevdw=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][12]
        generateResponse(prediction)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))

def getNNregression(credit_score,
                    first_time_homebuyer_flag,
                    no_unit,
                    occupancy_status,
                    orig_debttoincome,
                    orig_upb,
                    orig_loantovalue,
                    orig_interest_rate,
                    channel,
                    prepayment_penalty_mortgage_flag,
                    loan_purpose,
                    no_borrower):
    data = {
        "Inputs": {
            "input1":
                {
                    "ColumnNames": ["credit_score", "first_time_homebuyer_flag", "no_unit", "occupancy_status",
                                    "orig_debttoincome", "orig_upb", "orig_loantovalue", "orig_interest_rate",
                                    "channel", "prepayment_penalty_mortgage_flag", "loan_purpose", "no_borrower"],
                    "Values": [[credit_score,
                                first_time_homebuyer_flag,
                                no_unit,
                                occupancy_status,
                                orig_debttoincome,
                                orig_upb,
                                orig_loantovalue,
                                orig_interest_rate,
                                channel,
                                prepayment_penalty_mortgage_flag,
                                loan_purpose,
                                no_borrower]]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))
    url = 'https://ussouthcentral.services.azureml.net/workspaces/19e3e5bdfa6b42d381df6c8b26ce71f6/services/f884f02591d045609b44ae2119b85659/execute?api-version=2.0&details=true'
    api_key = 'sYJ0ewJLGX7Bw6xGEt63ndjPyz+vhNyICfd3GS5dKbaKikGyBY6m0mZzlhBa7xCm+ukgXqKfN5LruFzV1UEC6g=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][12]
        generateResponse(prediction)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))

def getLogisticRegression(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate):
    data = {
        "Inputs": {
            "input1":
                {
                    "ColumnNames": ["curr_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity",
                                    "curr_interest_rate"],
                    "Values": [[curr_loan_delinquency_status,loan_age,remaining_months_to_legal_maturity,curr_interest_rate] ]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/19e3e5bdfa6b42d381df6c8b26ce71f6/services/ab20fdcd3fdf45f195d510fa1c69f0b3/execute?api-version=2.0&details=true'
    api_key = 'sssMADI50IG25WsrZRfN8tpo0/Oao/KKcawwIeQeIW/RXyEB1E8cPMtytbIHzMxpwXE7WekyxdhTKI+gcwn1PQ=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][5]
	result ='You have '+  prediction + ' chance to be deliquent'
        generateResponse(result)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))

def getRFclassification(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate):
    data = {
        "Inputs": {
            "input1":
                {
                    "ColumnNames": ["curr_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity",
                                    "curr_interest_rate"],
                    "Values": [[curr_loan_delinquency_status, loan_age, remaining_months_to_legal_maturity,
                                curr_interest_rate]]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))
    url = 'https://ussouthcentral.services.azureml.net/workspaces/19e3e5bdfa6b42d381df6c8b26ce71f6/services/0b0aa613b81a471fb0b2419fe48ac628/execute?api-version=2.0&details=true'
    api_key = 'AtkHfN1H9Chh0RRw7xfVaHVVpNBtQrJmt0r5xxPjp52qv/Brdnn26CYVbw6IYHUnWXM3NHJk5pRd462rTBQD8Q=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][5]
        result ='You have '+  prediction + ' chance to be deliquent'
        generateResponse(result)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))

def getNNclassification(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate):
    data = {
        "Inputs": {
            "input1":
                {
                    "ColumnNames": ["curr_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity",
                                    "curr_interest_rate"],
                    "Values": [[curr_loan_delinquency_status, loan_age, remaining_months_to_legal_maturity,
                                curr_interest_rate]]
                }, },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))
    url = 'https://ussouthcentral.services.azureml.net/workspaces/19e3e5bdfa6b42d381df6c8b26ce71f6/services/6c40b4f93ac641f092430a8114bd901c/execute?api-version=2.0&details=true'
    api_key = 'gTCNaq6rhS4dKOdGEdZ0gDC3ewIWLCsMAUjWyqDdyYvi5M5eedLIWJwYwGXu48ZOmBJGkQcnEUAwgGM0Uy4qGA=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    req = urllib2.Request(url, body, headers)

    try:
        response = urllib2.urlopen(req)
        result = response.read()
        data = json.loads(result)
        prediction = data['Results']['output1']['value']['Values'][0][5]
	result ='You have '+  prediction + ' chance to be deliquent'
        generateResponse(result)
        return prediction

    except urllib2.HTTPError, error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read()))


def generateResponse(prediction):
    with open('/home/ubuntu/flaskapp/result.html','w') as myFile:
        myFile.write('<html>')
        myFile.write('<body>')
        myFile.write('<head>')
        myFile.write('The prediction result is:  ')
	myFile.write('</br>')
	myFile.write(prediction)
        myFile.write('</head></br>')
        myFile.write('</br>')
        myFile.write('</body>')
        myFile.write('</html>')

import urllib2
# If you are using Python 3+, import urllib instead of urllib2
import json


def getPrediction(orig_loantovalue,
                  credit_score,
                  orig_upb,
                  no_borrower,
                  orig_debttoincome,
                  no_unit,
                  orig_combined_loantovalue,
                  first_time_homebuyer_flag_Y,
                  occupancy_status_I,
                  occupancy_status_O,
                  occupancy_status_S,
                  channel_B,
                  channel_C,
                  channel_R,
                  channel_T,
                  prepayment_penalty_mortgage_flag_Y,
                  loan_purpose_C,
                  loan_purpose_N,
                  loan_purpose_P, ):
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
        myFile.write('The prediction result is:  ' + prediction)
        myFile.write('</head></br>')
        myFile.write('</br>')
        myFile.write('</body>')
        myFile.write('</html>')


import os.path
import ast
import prediction
# import linear_regression
import cnn_tweet

from flask import Flask, flash, Response, request, render_template

app = Flask(__name__)
app.secret_key = 'some_secret'

@app.route('/')
def hello_world():
    return 'Hello from Flask!'


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/prediction', methods=['GET'])
def getPrediction():  # pragma: no cover
    content = get_file('prediction.html')
    return Response(content, mimetype="text/html")


@app.route('/model', methods=['POST'])
def getModel():  # pragma: no cover
    print("asdfasdfasdf")
    model = request.form.get("optradio")
    content = get_file('prediction.html')
    if model == "linearregression":
        content = get_file('linearregression.html')
    elif model == "rfregression":
        content = get_file("rfregression.html")
    elif model == "nnregression":
        content = get_file("nnregression.html")
    elif model == "logisticregression":
        content = get_file("logisticregression.html")
    elif model == "rfclassification":
        content = get_file("rfclassification.html")
    elif model == "nnclassification":
        content = get_file("nnclassification.html")

    return Response(content, mimetype="text/html")

@app.route('/regression', methods=['GET'])
def metrics():  # pragma: no cover
    content = get_file('result.html')
    return Response(content, mimetype="text/html")


@app.route('/logisticregression', methods=['POST'])
def logisticregression():  # pragma: no cover
    curr_loan_delinquency_status = 0
    loan_age = ast.literal_eval(request.form.get("loan_age"))
    remaining_months_to_legal_maturity = ast.literal_eval(request.form.get("remaining_months_to_legal_maturity"))
    curr_interest_rate = ast.literal_eval(request.form.get("curr_interest_rate"))
    prediction.getLogisticRegression(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/rfclassification', methods=['POST'])
def rfclassification():  # pragma: no cover
    curr_loan_delinquency_status = 0
    loan_age = ast.literal_eval(request.form.get("loan_age"))
    remaining_months_to_legal_maturity = ast.literal_eval(request.form.get("remaining_months_to_legal_maturity"))
    curr_interest_rate = ast.literal_eval(request.form.get("curr_interest_rate"))
    prediction.getRFclassification(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/nnclassification', methods=['POST'])
def nnclassification():  # pragma: no cover
    curr_loan_delinquency_status = 0
    loan_age = ast.literal_eval(request.form.get("loan_age"))
    remaining_months_to_legal_maturity = ast.literal_eval(request.form.get("remaining_months_to_legal_maturity"))
    curr_interest_rate = ast.literal_eval(request.form.get("curr_interest_rate"))
    prediction.getNNclassification(curr_loan_delinquency_status,
                                     loan_age,
                                     remaining_months_to_legal_maturity,
                                     curr_interest_rate)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/linearregression', methods=['POST'])
def linearregression():  # pragma: no cover
    orig_loantovalue = ast.literal_eval(request.form.get("orig_loantovalue"))
    credit_score = ast.literal_eval(request.form.get("credit_score"))
    orig_upb = ast.literal_eval(request.form.get("orig_upb"))
    no_borrower = ast.literal_eval(request.form.get("no_borrower"))
    orig_debttoincome = ast.literal_eval(request.form.get("orig_debttoincome"))
    no_unit = ast.literal_eval(request.form.get("no_unit"))
    orig_combined_loantovalue = ast.literal_eval(request.form.get("orig_combined_loantovalue"))

    first_time_homebuyer_flag_Y = 0.0
    occupancy_status_I = 0.0
    occupancy_status_O = 0.0
    occupancy_status_S = 0.0
    channel_B = 0.0
    channel_C = 0.0
    channel_R = 0.0
    channel_T = 0.0
    prepayment_penalty_mortgage_flag_Y = 0.0
    loan_purpose_C = 0.0
    loan_purpose_N = 0.0
    loan_purpose_P = 0.0

    first_time_homebuyer_flag = request.form.get("first_time_homebuyer_flag")
    occupancy_status = request.form.get("occupancy_status")
    channel = request.form.get("channel")
    prepayment_penalty_mortgage_flag = request.form.get("prepayment_penalty_mortgage_flag")
    loan_purpose = request.form.get("loan_purpose")
    if first_time_homebuyer_flag == "Y":
        first_time_homebuyer_flag_Y = 1.0

    if occupancy_status == "I":
        occupancy_status_I = 1.0
    elif occupancy_status == "O":
        occupancy_status_O = 1.0
    elif occupancy_status == "S":
        occupancy_status_S = 1.0

    if channel == "B":
        channel_B = 1.0
    elif channel == "C":
        channel_C = 1.0
    elif channel == "R":
        channel_R = 1.0
    elif channel == "T":
        channel_T = 1.0

    if prepayment_penalty_mortgage_flag == "Y":
        prepayment_penalty_mortgage_flag_Y = 1.0

    if loan_purpose == "C":
        loan_purpose_C = 1.0
    elif loan_purpose == "N":
        loan_purpose_N = 1.0
    elif loan_purpose == "P":
        loan_purpose_P = 1.0

    prediction.getLinearRegression(orig_loantovalue,
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
                                    loan_purpose_P)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/rfregression', methods=['POST'])
def rfregression():  # pragma: no cover
    credit_score = ast.literal_eval(request.form.get("credit_score"))
    no_unit = ast.literal_eval(request.form.get("no_unit"))
    orig_debttoincome = ast.literal_eval(request.form.get("orig_debttoincome"))
    orig_upb = ast.literal_eval(request.form.get("orig_upb"))
    orig_loantovalue = ast.literal_eval(request.form.get("orig_loantovalue"))
    orig_interest_rate = 0
    no_borrower = ast.literal_eval(request.form.get("no_borrower"))

    first_time_homebuyer_flag = request.form.get("first_time_homebuyer_flag")
    occupancy_status = request.form.get("occupancy_status")
    channel = request.form.get("channel")
    prepayment_penalty_mortgage_flag = request.form.get("prepayment_penalty_mortgage_flag")
    loan_purpose = request.form.get("loan_purpose")
    prediction.getRFregression(credit_score,
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
                    no_borrower)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/nnregression', methods=['POST'])
def nnregression():  # pragma: no cover
    credit_score = ast.literal_eval(request.form.get("credit_score"))
    no_unit = ast.literal_eval(request.form.get("no_unit"))
    orig_debttoincome = ast.literal_eval(request.form.get("orig_debttoincome"))
    orig_upb = ast.literal_eval(request.form.get("orig_upb"))
    orig_loantovalue = ast.literal_eval(request.form.get("orig_loantovalue"))
    orig_interest_rate = 0
    no_borrower = ast.literal_eval(request.form.get("no_borrower"))

    first_time_homebuyer_flag = request.form.get("first_time_homebuyer_flag")
    occupancy_status = request.form.get("occupancy_status")
    channel = request.form.get("channel")
    prepayment_penalty_mortgage_flag = request.form.get("prepayment_penalty_mortgage_flag")
    loan_purpose = request.form.get("loan_purpose")
    prediction.getNNregression(credit_score,
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
                    no_borrower)
    content = get_file("result.html")
    return Response(content, mimetype="text/html")

@app.route('/cnn-tweet', methods=['GET'])
def getCnnTweet():  # pragma: no cover
    return render_template('cnn_tweet.html')
	
@app.route('/cnn-tweet-label', methods=['POST'])
def postCnnTweet():  # pragma: no cover
    sentence = request.form["sentence"]
    label = cnn_tweet.getClassfication(sentence)
	
    flash("Result: {:d}".format(label))
    return render_template('cnn_tweet.html', sentence=sentence)
	
@app.route('/cnn-tweet-file', methods=['GET'])
def getCnnTweetFile():  # pragma: no cover
    return render_template('cnn_tweet_file.html')
	
@app.route('/cnn-tweet-file-label', methods=['POST'])
def postCnnTweetFile():  # pragma: no cover
    cnn_tweet.getTestDataOutput()
	
    # flash("Result: {:d}".format(label))
    return render_template('cnn_tweet_file.html')
	
if __name__ == '__main__':
    app.run()


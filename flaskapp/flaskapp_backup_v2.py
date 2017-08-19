import os.path
import ast
import linear_regression

from flask import Flask, Response, request

app = Flask(__name__)


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


@app.route('/regression', methods=['GET'])
def metrics():  # pragma: no cover
    content = get_file('regression.html')
    return Response(content, mimetype="text/html")


@app.route('/returnlocations', methods=['POST'])
def returnlocations():  # pragma: no cover
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

    linear_regression.getPrediction(orig_loantovalue,
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
    content = get_file("prediction.html")
    return Response(content, mimetype="text/html")


if __name__ == '__main__':
    app.run()


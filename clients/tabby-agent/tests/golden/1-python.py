import datetime

def parse_expenses(expenses_string):
    """Parse the list of expenses and return the list of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    Example expenses_string:
        2016-01-02 -34.01 USD
        2016-01-03 2.59 DKK
        2016-01-03 -2.72 EUR
    """
    for line in expenses_string.split('\\n'):
        ⏩⏭if line.startswith('#'):
            continue
        date, value, currency = line.split(' - ')
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        yield date, float(value), currency⏮⏪
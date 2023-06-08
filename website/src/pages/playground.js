import React from 'react';
import Layout from '@theme/Layout';
import Monaco from "@site/src/components/Monaco";

export default function Home() {
  return (
    <Layout title="Playground">
      <main className="flex-grow flex relative">
        <div className="Playground absolute inset-0">
          <Monaco
            defaultValue={DEFAULT_CODE}
            options={{
              automaticLayout: true,
              fontSize: 16,
            }}
          />
        </div>
      </main>
    </Layout>
  );
}

const DEFAULT_CODE = `import datetime

def parse_expenses(expenses_string):
    """Parse the list of expenses and return the list of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    Example expenses_string:
        2016-01-02 -34.01 USD
        2016-01-03 2.59 DKK
        2016-01-03 -2.72 EUR
    """
    for line in expenses_string.split('\\n'):`

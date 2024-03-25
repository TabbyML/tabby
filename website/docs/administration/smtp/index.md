---
sidebar_position: 5
---

# Mail Delivery

Tabby uses an SMTP server of your choice to send emails. Some functionaties like password reset, email notifications, etc. require an SMTP server to be configured.

You can configure the SMTP server settings in the **Mail Delivery** page.

 ## Configuring SMTP via Amazon SES
 
 To use Amazon SES, first [follow these steps to creating and verifying identities](https://docs.aws.amazon.com/ses/latest/dg/creating-identities.html). 
 Then, use [AWS Access Management(IAM)](https://aws.amazon.com/iam/) to create an SMTP credential.
 Once you have an IAM user with the necessary permissions, you can use the credentials to configure Tabby like below:

 ![Amazon SES](./ses.png)

## Configuring other SMTP providers

Other providers such as [SendGrid](https://sendgrid.com/), [Mailgun](https://www.mailgun.com/) or [Resend](https://resend.com) can be configured by providing the SMTP server details. You can find the SMTP server details in the respective provider's documentation.



## Send a Test Email

To verify email sending is working correctly, fill in the **Send Test Email To** field and click **Send** button, Tabby will send a test email using your SMTP configuration. If everything is correct, you will receive a mail like:

![Test Email](./test-email.png)
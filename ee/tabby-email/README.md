## Add new email template for Tabby

- Navigate to `ee/tabby-email/emails`
- Run `yarn && yarn dev`
- This will start a webserver and show a link where you can preview emails
- Copy existing email templates to create new emails
- Run `make update-email-templates` to generate html files from your react email
- Go to `ee/tabby-webserver/src/service/email/templates.rs` and add a new function for your template
- Add a function in the `EmailService` trait to send your new email and implement it in `ee/tabby-webserver/src/service/email/mod.rs`

---
title: "New in Tabby: Code Browser"
authors: []
tags: []
---

![The New Tabby Code Browser](./code-browser-new.png)

We’re excited to launch the Code Browser, a new feature that enhances your ability to navigate and understand code within your projects. Accessible from your dashboard, it uses Tabby's machine-learning capabilities to streamline code exploration by integrating both public and private insights directly in your workflow.

<!-- Available in version 0.12.3. See our upgrade guide -->

![A diagram of the "Code Browser" link on the dashboard](./dashboard-link.png)

## What it does

The Code Browser adds functions to generate from and explain any code in your repo. Navigate to a file and highlight some text to get started:

![Tabby's "explain" and "generate" functions](./explain-and-generate.png)

**Generate** provides options to create documentation or unit tests, while **explain** will walk you through the code step by step. And if you need something clarified or tweaked, just follow up with Tabby to continue the conversation.

Any repository connected to Tabby is automatically indexed for browsing. You can connect providers like GitHub, or use a self-host option to ensure total data control. You can also configure your LLM and integrate securely with tools like Jira and Linear for organizational insights.

Check out the [repository docs](https://tabby.tabbyml.com/docs/administration/context/) to learn about connecting repos.

## What’s next?

The Tabby train doesn’t stop moving. We’re already working on enhancements, such as:

- **Linkable answers**. Chats that can be shared from and viewed in the Tabby interface.
- **Improved usability**. One-click actions as well as IDE integrations of generate and explain.
- **Branch scoping**. Browse by branch in addition to repo. If a branch is associated with a PR, its diffs will be highlighted for easy comparisons.
- **Refactor**. Work with Tabby to improve a function or file.

Be sure to follow Tabby on [GitHub](https://github.com/TabbyML/tabby) for future updates and announcements. And as always, we welcome your feedback and suggestions on our [Slack channel](https://slack.tabbyml.com/).

Until next time, happy browsing!

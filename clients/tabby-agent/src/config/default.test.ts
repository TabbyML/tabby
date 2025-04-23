import { expect } from "chai";
import { stringToRegExp } from "../utils/string";
import { defaultConfigData } from "./default";

describe("Config: generateCommitMessage.responseMatcher", () => {
  // Test parameters
  const responseMatcher = defaultConfigData.chat.generateCommitMessage.responseMatcher;
  const regExp = stringToRegExp(responseMatcher);

  // Helper function for reusing test logic
  function testResponseMatch(testCase: string, input: string, expectedMatch: string) {
    it(testCase, () => {
      const match = regExp.exec(input);
      expect(match).to.not.be.null;
      if (match) {
        expect(match[1]).to.equal(expectedMatch);
      }
    });
  }

  // Core functionality: Extract commit message from simple response
  testResponseMatch(
    "test for extracting conventional commit message from simple response",
    "Based on the diff, I would suggest the following commit message:\n\nfeat(core): add new feature\n",
    "feat(core): add new feature",
  );

  // Core functionality: Handle commit messages in code blocks
  testResponseMatch(
    "test for handling responses with code blocks",
    `Based on the diff, here's an appropriate commit message:

\`\`\`
fix(auth): resolve user authentication timeout issue
\`\`\`

This commit message follows the conventional format with a 'fix' type and 'auth' scope.`,
    "fix(auth): resolve user authentication timeout issue",
  );

  // Bug fix: Handle commit messages with double quotes
  testResponseMatch(
    "test for handling responses with double quotes",
    `Based on the provided diff, I recommend using the following commit message:

"docs(readme): update installation instructions"

This commit message follows the conventional format and accurately describes the changes made to the documentation.`,
    "docs(readme): update installation instructions",
  );

  // Bug fix: Handle commit messages with single quotes and backticks
  testResponseMatch(
    "test for handling responses with single quotes and backticks",
    `Here are some commit message options:
'chore(build): update dependencies'
\`test(components): add unit tests for login form\``,
    "chore(build): update dependencies",
  );

  // Bug fix: Handle responses with markdown images
  testResponseMatch(
    "test for handling responses with markdown images",
    `Here's a diagram showing the changes you made:
![Diagram](https://example.com/diagram.png)

Based on the diff, I suggest the following commit message:

feat(ui): improve button design and layout

[Diagram]: https://example.com/diagram-ref.png`,
    "feat(ui): improve button design and layout",
  );
});

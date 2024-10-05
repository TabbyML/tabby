const packageJson = require("../package.json");
const semver = require("semver");

const minor = semver.minor(packageJson.version);

if (minor % 2 === 0) {
  console.warn("Even minor version, release as stable channel");
  console.log("publish");
} else {
  console.warn("Odd minor version, release as prerelease channel");
  console.log("publish-prerelease");
}

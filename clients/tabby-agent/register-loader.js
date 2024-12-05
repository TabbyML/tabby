const path = require("path");
const fs = require("fs");
const REQUIRE_PATH_TEST = /\.md$/;

function register() {
  const Module = require("module");
  const originalLoad = Module._load;
  const cwd = process.cwd();
  Module._load = function (request, _parent) {
    if (request.match(REQUIRE_PATH_TEST)) {
      return fs.readFileSync(path.join(path.dirname(_parent ? _parent.filename : cwd), request), "utf8");
    }
    return originalLoad.apply(this, arguments);
  };

  return () => {
    Module._load = originalLoad;
  };
}

register();

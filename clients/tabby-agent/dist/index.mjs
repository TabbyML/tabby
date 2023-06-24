var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __require = /* @__PURE__ */ ((x4) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x4, {
  get: (a7, b5) => (typeof require !== "undefined" ? require : a7)[b5]
}) : x4)(function(x4) {
  if (typeof require !== "undefined")
    return require.apply(this, arguments);
  throw new Error('Dynamic require of "' + x4 + '" is not supported');
});
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __commonJS = (cb, mod) => function __require2() {
  return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __export = (target, all3) => {
  for (var name3 in all3)
    __defProp(target, name3, { get: all3[name3], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
var __accessCheck = (obj, member, msg) => {
  if (!member.has(obj))
    throw TypeError("Cannot " + msg);
};
var __privateGet = (obj, member, getter) => {
  __accessCheck(obj, member, "read from private field");
  return getter ? getter.call(obj) : member.get(obj);
};
var __privateAdd = (obj, member, value) => {
  if (member.has(obj))
    throw TypeError("Cannot add the same private member more than once");
  member instanceof WeakSet ? member.add(obj) : member.set(obj, value);
};
var __privateSet = (obj, member, value, setter) => {
  __accessCheck(obj, member, "write to private field");
  setter ? setter.call(obj, value) : member.set(obj, value);
  return value;
};
var __privateWrapper = (obj, member, setter, getter) => ({
  set _(value) {
    __privateSet(obj, member, value, setter);
  },
  get _() {
    return __privateGet(obj, member, getter);
  }
});
var __privateMethod = (obj, member, method) => {
  __accessCheck(obj, member, "access private method");
  return method;
};

// node_modules/esbuild-plugin-polyfill-node/polyfills/global.js
var global;
var init_global = __esm({
  "node_modules/esbuild-plugin-polyfill-node/polyfills/global.js"() {
    global = globalThis;
  }
});

// node_modules/esbuild-plugin-polyfill-node/polyfills/__dirname.js
var init_dirname = __esm({
  "node_modules/esbuild-plugin-polyfill-node/polyfills/__dirname.js"() {
  }
});

// node_modules/esbuild-plugin-polyfill-node/polyfills/__filename.js
var init_filename = __esm({
  "node_modules/esbuild-plugin-polyfill-node/polyfills/__filename.js"() {
  }
});

// node_modules/@jspm/core/nodelibs/browser/process.js
var process_exports = {};
__export(process_exports, {
  _debugEnd: () => _debugEnd,
  _debugProcess: () => _debugProcess,
  _events: () => _events,
  _eventsCount: () => _eventsCount,
  _exiting: () => _exiting,
  _fatalExceptions: () => _fatalExceptions,
  _getActiveHandles: () => _getActiveHandles,
  _getActiveRequests: () => _getActiveRequests,
  _kill: () => _kill,
  _linkedBinding: () => _linkedBinding,
  _maxListeners: () => _maxListeners,
  _preload_modules: () => _preload_modules,
  _rawDebug: () => _rawDebug,
  _startProfilerIdleNotifier: () => _startProfilerIdleNotifier,
  _stopProfilerIdleNotifier: () => _stopProfilerIdleNotifier,
  _tickCallback: () => _tickCallback,
  abort: () => abort,
  addListener: () => addListener,
  allowedNodeEnvironmentFlags: () => allowedNodeEnvironmentFlags,
  arch: () => arch,
  argv: () => argv,
  argv0: () => argv0,
  assert: () => assert,
  binding: () => binding,
  chdir: () => chdir,
  config: () => config,
  cpuUsage: () => cpuUsage,
  cwd: () => cwd,
  debugPort: () => debugPort,
  default: () => process,
  dlopen: () => dlopen,
  domain: () => domain,
  emit: () => emit,
  emitWarning: () => emitWarning,
  env: () => env,
  execArgv: () => execArgv,
  execPath: () => execPath,
  exit: () => exit,
  features: () => features,
  hasUncaughtExceptionCaptureCallback: () => hasUncaughtExceptionCaptureCallback,
  hrtime: () => hrtime,
  kill: () => kill,
  listeners: () => listeners,
  memoryUsage: () => memoryUsage,
  moduleLoadList: () => moduleLoadList,
  nextTick: () => nextTick,
  off: () => off,
  on: () => on,
  once: () => once,
  openStdin: () => openStdin,
  pid: () => pid,
  platform: () => platform,
  ppid: () => ppid,
  prependListener: () => prependListener,
  prependOnceListener: () => prependOnceListener,
  reallyExit: () => reallyExit,
  release: () => release,
  removeAllListeners: () => removeAllListeners,
  removeListener: () => removeListener,
  resourceUsage: () => resourceUsage,
  setSourceMapsEnabled: () => setSourceMapsEnabled,
  setUncaughtExceptionCaptureCallback: () => setUncaughtExceptionCaptureCallback,
  stderr: () => stderr,
  stdin: () => stdin,
  stdout: () => stdout,
  title: () => title,
  umask: () => umask,
  uptime: () => uptime,
  version: () => version,
  versions: () => versions
});
function unimplemented(name3) {
  throw new Error("Node.js process " + name3 + " is not supported by JSPM core outside of Node.js");
}
function cleanUpNextTick() {
  if (!draining || !currentQueue)
    return;
  draining = false;
  if (currentQueue.length) {
    queue = currentQueue.concat(queue);
  } else {
    queueIndex = -1;
  }
  if (queue.length)
    drainQueue();
}
function drainQueue() {
  if (draining)
    return;
  var timeout = setTimeout(cleanUpNextTick, 0);
  draining = true;
  var len = queue.length;
  while (len) {
    currentQueue = queue;
    queue = [];
    while (++queueIndex < len) {
      if (currentQueue)
        currentQueue[queueIndex].run();
    }
    queueIndex = -1;
    len = queue.length;
  }
  currentQueue = null;
  draining = false;
  clearTimeout(timeout);
}
function nextTick(fun) {
  var args = new Array(arguments.length - 1);
  if (arguments.length > 1) {
    for (var i7 = 1; i7 < arguments.length; i7++)
      args[i7 - 1] = arguments[i7];
  }
  queue.push(new Item(fun, args));
  if (queue.length === 1 && !draining)
    setTimeout(drainQueue, 0);
}
function Item(fun, array) {
  this.fun = fun;
  this.array = array;
}
function noop() {
}
function _linkedBinding(name3) {
  unimplemented("_linkedBinding");
}
function dlopen(name3) {
  unimplemented("dlopen");
}
function _getActiveRequests() {
  return [];
}
function _getActiveHandles() {
  return [];
}
function assert(condition, message) {
  if (!condition)
    throw new Error(message || "assertion error");
}
function hasUncaughtExceptionCaptureCallback() {
  return false;
}
function uptime() {
  return _performance.now() / 1e3;
}
function hrtime(previousTimestamp) {
  var baseNow = Math.floor((Date.now() - _performance.now()) * 1e-3);
  var clocktime = _performance.now() * 1e-3;
  var seconds = Math.floor(clocktime) + baseNow;
  var nanoseconds = Math.floor(clocktime % 1 * 1e9);
  if (previousTimestamp) {
    seconds = seconds - previousTimestamp[0];
    nanoseconds = nanoseconds - previousTimestamp[1];
    if (nanoseconds < 0) {
      seconds--;
      nanoseconds += nanoPerSec;
    }
  }
  return [seconds, nanoseconds];
}
function on() {
  return process;
}
function listeners(name3) {
  return [];
}
var queue, draining, currentQueue, queueIndex, title, arch, platform, env, argv, execArgv, version, versions, emitWarning, binding, umask, cwd, chdir, release, _rawDebug, moduleLoadList, domain, _exiting, config, reallyExit, _kill, cpuUsage, resourceUsage, memoryUsage, kill, exit, openStdin, allowedNodeEnvironmentFlags, features, _fatalExceptions, setUncaughtExceptionCaptureCallback, _tickCallback, _debugProcess, _debugEnd, _startProfilerIdleNotifier, _stopProfilerIdleNotifier, stdout, stderr, stdin, abort, pid, ppid, execPath, debugPort, argv0, _preload_modules, setSourceMapsEnabled, _performance, nowOffset, nanoPerSec, _maxListeners, _events, _eventsCount, addListener, once, off, removeListener, removeAllListeners, emit, prependListener, prependOnceListener, process;
var init_process = __esm({
  "node_modules/@jspm/core/nodelibs/browser/process.js"() {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    queue = [];
    draining = false;
    queueIndex = -1;
    Item.prototype.run = function() {
      this.fun.apply(null, this.array);
    };
    title = "browser";
    arch = "x64";
    platform = "browser";
    env = {
      PATH: "/usr/bin",
      LANG: navigator.language + ".UTF-8",
      PWD: "/",
      HOME: "/home",
      TMP: "/tmp"
    };
    argv = ["/usr/bin/node"];
    execArgv = [];
    version = "v16.8.0";
    versions = {};
    emitWarning = function(message, type2) {
      console.warn((type2 ? type2 + ": " : "") + message);
    };
    binding = function(name3) {
      unimplemented("binding");
    };
    umask = function(mask) {
      return 0;
    };
    cwd = function() {
      return "/";
    };
    chdir = function(dir) {
    };
    release = {
      name: "node",
      sourceUrl: "",
      headersUrl: "",
      libUrl: ""
    };
    _rawDebug = noop;
    moduleLoadList = [];
    domain = {};
    _exiting = false;
    config = {};
    reallyExit = noop;
    _kill = noop;
    cpuUsage = function() {
      return {};
    };
    resourceUsage = cpuUsage;
    memoryUsage = cpuUsage;
    kill = noop;
    exit = noop;
    openStdin = noop;
    allowedNodeEnvironmentFlags = {};
    features = {
      inspector: false,
      debug: false,
      uv: false,
      ipv6: false,
      tls_alpn: false,
      tls_sni: false,
      tls_ocsp: false,
      tls: false,
      cached_builtins: true
    };
    _fatalExceptions = noop;
    setUncaughtExceptionCaptureCallback = noop;
    _tickCallback = noop;
    _debugProcess = noop;
    _debugEnd = noop;
    _startProfilerIdleNotifier = noop;
    _stopProfilerIdleNotifier = noop;
    stdout = void 0;
    stderr = void 0;
    stdin = void 0;
    abort = noop;
    pid = 2;
    ppid = 1;
    execPath = "/bin/usr/node";
    debugPort = 9229;
    argv0 = "node";
    _preload_modules = [];
    setSourceMapsEnabled = noop;
    _performance = {
      now: typeof performance !== "undefined" ? performance.now.bind(performance) : void 0,
      timing: typeof performance !== "undefined" ? performance.timing : void 0
    };
    if (_performance.now === void 0) {
      nowOffset = Date.now();
      if (_performance.timing && _performance.timing.navigationStart) {
        nowOffset = _performance.timing.navigationStart;
      }
      _performance.now = () => Date.now() - nowOffset;
    }
    nanoPerSec = 1e9;
    hrtime.bigint = function(time) {
      var diff = hrtime(time);
      if (typeof BigInt === "undefined") {
        return diff[0] * nanoPerSec + diff[1];
      }
      return BigInt(diff[0] * nanoPerSec) + BigInt(diff[1]);
    };
    _maxListeners = 10;
    _events = {};
    _eventsCount = 0;
    addListener = on;
    once = on;
    off = on;
    removeListener = on;
    removeAllListeners = on;
    emit = noop;
    prependListener = on;
    prependOnceListener = on;
    process = {
      version,
      versions,
      arch,
      platform,
      release,
      _rawDebug,
      moduleLoadList,
      binding,
      _linkedBinding,
      _events,
      _eventsCount,
      _maxListeners,
      on,
      addListener,
      once,
      off,
      removeListener,
      removeAllListeners,
      emit,
      prependListener,
      prependOnceListener,
      listeners,
      domain,
      _exiting,
      config,
      dlopen,
      uptime,
      _getActiveRequests,
      _getActiveHandles,
      reallyExit,
      _kill,
      cpuUsage,
      resourceUsage,
      memoryUsage,
      kill,
      exit,
      openStdin,
      allowedNodeEnvironmentFlags,
      assert,
      features,
      _fatalExceptions,
      setUncaughtExceptionCaptureCallback,
      hasUncaughtExceptionCaptureCallback,
      emitWarning,
      nextTick,
      _tickCallback,
      _debugProcess,
      _debugEnd,
      _startProfilerIdleNotifier,
      _stopProfilerIdleNotifier,
      stdout,
      stdin,
      stderr,
      abort,
      umask,
      chdir,
      cwd,
      env,
      title,
      argv,
      execArgv,
      pid,
      ppid,
      execPath,
      debugPort,
      hrtime,
      argv0,
      _preload_modules,
      setSourceMapsEnabled
    };
  }
});

// node_modules/esbuild-plugin-polyfill-node/polyfills/process.js
var init_process2 = __esm({
  "node_modules/esbuild-plugin-polyfill-node/polyfills/process.js"() {
    init_process();
  }
});

// node_modules/@jspm/core/nodelibs/browser/buffer.js
function dew$2() {
  if (_dewExec$2)
    return exports$3;
  _dewExec$2 = true;
  exports$3.byteLength = byteLength;
  exports$3.toByteArray = toByteArray;
  exports$3.fromByteArray = fromByteArray;
  var lookup = [];
  var revLookup = [];
  var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
  var code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (var i7 = 0, len = code.length; i7 < len; ++i7) {
    lookup[i7] = code[i7];
    revLookup[code.charCodeAt(i7)] = i7;
  }
  revLookup["-".charCodeAt(0)] = 62;
  revLookup["_".charCodeAt(0)] = 63;
  function getLens(b64) {
    var len2 = b64.length;
    if (len2 % 4 > 0) {
      throw new Error("Invalid string. Length must be a multiple of 4");
    }
    var validLen = b64.indexOf("=");
    if (validLen === -1)
      validLen = len2;
    var placeHoldersLen = validLen === len2 ? 0 : 4 - validLen % 4;
    return [validLen, placeHoldersLen];
  }
  function byteLength(b64) {
    var lens = getLens(b64);
    var validLen = lens[0];
    var placeHoldersLen = lens[1];
    return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
  }
  function _byteLength(b64, validLen, placeHoldersLen) {
    return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
  }
  function toByteArray(b64) {
    var tmp;
    var lens = getLens(b64);
    var validLen = lens[0];
    var placeHoldersLen = lens[1];
    var arr = new Arr(_byteLength(b64, validLen, placeHoldersLen));
    var curByte = 0;
    var len2 = placeHoldersLen > 0 ? validLen - 4 : validLen;
    var i8;
    for (i8 = 0; i8 < len2; i8 += 4) {
      tmp = revLookup[b64.charCodeAt(i8)] << 18 | revLookup[b64.charCodeAt(i8 + 1)] << 12 | revLookup[b64.charCodeAt(i8 + 2)] << 6 | revLookup[b64.charCodeAt(i8 + 3)];
      arr[curByte++] = tmp >> 16 & 255;
      arr[curByte++] = tmp >> 8 & 255;
      arr[curByte++] = tmp & 255;
    }
    if (placeHoldersLen === 2) {
      tmp = revLookup[b64.charCodeAt(i8)] << 2 | revLookup[b64.charCodeAt(i8 + 1)] >> 4;
      arr[curByte++] = tmp & 255;
    }
    if (placeHoldersLen === 1) {
      tmp = revLookup[b64.charCodeAt(i8)] << 10 | revLookup[b64.charCodeAt(i8 + 1)] << 4 | revLookup[b64.charCodeAt(i8 + 2)] >> 2;
      arr[curByte++] = tmp >> 8 & 255;
      arr[curByte++] = tmp & 255;
    }
    return arr;
  }
  function tripletToBase64(num) {
    return lookup[num >> 18 & 63] + lookup[num >> 12 & 63] + lookup[num >> 6 & 63] + lookup[num & 63];
  }
  function encodeChunk(uint8, start, end) {
    var tmp;
    var output = [];
    for (var i8 = start; i8 < end; i8 += 3) {
      tmp = (uint8[i8] << 16 & 16711680) + (uint8[i8 + 1] << 8 & 65280) + (uint8[i8 + 2] & 255);
      output.push(tripletToBase64(tmp));
    }
    return output.join("");
  }
  function fromByteArray(uint8) {
    var tmp;
    var len2 = uint8.length;
    var extraBytes = len2 % 3;
    var parts = [];
    var maxChunkLength = 16383;
    for (var i8 = 0, len22 = len2 - extraBytes; i8 < len22; i8 += maxChunkLength) {
      parts.push(encodeChunk(uint8, i8, i8 + maxChunkLength > len22 ? len22 : i8 + maxChunkLength));
    }
    if (extraBytes === 1) {
      tmp = uint8[len2 - 1];
      parts.push(lookup[tmp >> 2] + lookup[tmp << 4 & 63] + "==");
    } else if (extraBytes === 2) {
      tmp = (uint8[len2 - 2] << 8) + uint8[len2 - 1];
      parts.push(lookup[tmp >> 10] + lookup[tmp >> 4 & 63] + lookup[tmp << 2 & 63] + "=");
    }
    return parts.join("");
  }
  return exports$3;
}
function dew$1() {
  if (_dewExec$1)
    return exports$2;
  _dewExec$1 = true;
  exports$2.read = function(buffer2, offset, isLE, mLen, nBytes) {
    var e10, m6;
    var eLen = nBytes * 8 - mLen - 1;
    var eMax = (1 << eLen) - 1;
    var eBias = eMax >> 1;
    var nBits = -7;
    var i7 = isLE ? nBytes - 1 : 0;
    var d6 = isLE ? -1 : 1;
    var s6 = buffer2[offset + i7];
    i7 += d6;
    e10 = s6 & (1 << -nBits) - 1;
    s6 >>= -nBits;
    nBits += eLen;
    for (; nBits > 0; e10 = e10 * 256 + buffer2[offset + i7], i7 += d6, nBits -= 8) {
    }
    m6 = e10 & (1 << -nBits) - 1;
    e10 >>= -nBits;
    nBits += mLen;
    for (; nBits > 0; m6 = m6 * 256 + buffer2[offset + i7], i7 += d6, nBits -= 8) {
    }
    if (e10 === 0) {
      e10 = 1 - eBias;
    } else if (e10 === eMax) {
      return m6 ? NaN : (s6 ? -1 : 1) * Infinity;
    } else {
      m6 = m6 + Math.pow(2, mLen);
      e10 = e10 - eBias;
    }
    return (s6 ? -1 : 1) * m6 * Math.pow(2, e10 - mLen);
  };
  exports$2.write = function(buffer2, value, offset, isLE, mLen, nBytes) {
    var e10, m6, c7;
    var eLen = nBytes * 8 - mLen - 1;
    var eMax = (1 << eLen) - 1;
    var eBias = eMax >> 1;
    var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
    var i7 = isLE ? 0 : nBytes - 1;
    var d6 = isLE ? 1 : -1;
    var s6 = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
    value = Math.abs(value);
    if (isNaN(value) || value === Infinity) {
      m6 = isNaN(value) ? 1 : 0;
      e10 = eMax;
    } else {
      e10 = Math.floor(Math.log(value) / Math.LN2);
      if (value * (c7 = Math.pow(2, -e10)) < 1) {
        e10--;
        c7 *= 2;
      }
      if (e10 + eBias >= 1) {
        value += rt / c7;
      } else {
        value += rt * Math.pow(2, 1 - eBias);
      }
      if (value * c7 >= 2) {
        e10++;
        c7 /= 2;
      }
      if (e10 + eBias >= eMax) {
        m6 = 0;
        e10 = eMax;
      } else if (e10 + eBias >= 1) {
        m6 = (value * c7 - 1) * Math.pow(2, mLen);
        e10 = e10 + eBias;
      } else {
        m6 = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
        e10 = 0;
      }
    }
    for (; mLen >= 8; buffer2[offset + i7] = m6 & 255, i7 += d6, m6 /= 256, mLen -= 8) {
    }
    e10 = e10 << mLen | m6;
    eLen += mLen;
    for (; eLen > 0; buffer2[offset + i7] = e10 & 255, i7 += d6, e10 /= 256, eLen -= 8) {
    }
    buffer2[offset + i7 - d6] |= s6 * 128;
  };
  return exports$2;
}
function dew() {
  if (_dewExec)
    return exports$1;
  _dewExec = true;
  const base642 = dew$2();
  const ieee754 = dew$1();
  const customInspectSymbol = typeof Symbol === "function" && typeof Symbol["for"] === "function" ? Symbol["for"]("nodejs.util.inspect.custom") : null;
  exports$1.Buffer = Buffer3;
  exports$1.SlowBuffer = SlowBuffer;
  exports$1.INSPECT_MAX_BYTES = 50;
  const K_MAX_LENGTH = 2147483647;
  exports$1.kMaxLength = K_MAX_LENGTH;
  Buffer3.TYPED_ARRAY_SUPPORT = typedArraySupport();
  if (!Buffer3.TYPED_ARRAY_SUPPORT && typeof console !== "undefined" && typeof console.error === "function") {
    console.error("This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support.");
  }
  function typedArraySupport() {
    try {
      const arr = new Uint8Array(1);
      const proto = {
        foo: function() {
          return 42;
        }
      };
      Object.setPrototypeOf(proto, Uint8Array.prototype);
      Object.setPrototypeOf(arr, proto);
      return arr.foo() === 42;
    } catch (e10) {
      return false;
    }
  }
  Object.defineProperty(Buffer3.prototype, "parent", {
    enumerable: true,
    get: function() {
      if (!Buffer3.isBuffer(this))
        return void 0;
      return this.buffer;
    }
  });
  Object.defineProperty(Buffer3.prototype, "offset", {
    enumerable: true,
    get: function() {
      if (!Buffer3.isBuffer(this))
        return void 0;
      return this.byteOffset;
    }
  });
  function createBuffer(length) {
    if (length > K_MAX_LENGTH) {
      throw new RangeError('The value "' + length + '" is invalid for option "size"');
    }
    const buf = new Uint8Array(length);
    Object.setPrototypeOf(buf, Buffer3.prototype);
    return buf;
  }
  function Buffer3(arg, encodingOrOffset, length) {
    if (typeof arg === "number") {
      if (typeof encodingOrOffset === "string") {
        throw new TypeError('The "string" argument must be of type string. Received type number');
      }
      return allocUnsafe(arg);
    }
    return from(arg, encodingOrOffset, length);
  }
  Buffer3.poolSize = 8192;
  function from(value, encodingOrOffset, length) {
    if (typeof value === "string") {
      return fromString(value, encodingOrOffset);
    }
    if (ArrayBuffer.isView(value)) {
      return fromArrayView(value);
    }
    if (value == null) {
      throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value);
    }
    if (isInstance(value, ArrayBuffer) || value && isInstance(value.buffer, ArrayBuffer)) {
      return fromArrayBuffer(value, encodingOrOffset, length);
    }
    if (typeof SharedArrayBuffer !== "undefined" && (isInstance(value, SharedArrayBuffer) || value && isInstance(value.buffer, SharedArrayBuffer))) {
      return fromArrayBuffer(value, encodingOrOffset, length);
    }
    if (typeof value === "number") {
      throw new TypeError('The "value" argument must not be of type number. Received type number');
    }
    const valueOf = value.valueOf && value.valueOf();
    if (valueOf != null && valueOf !== value) {
      return Buffer3.from(valueOf, encodingOrOffset, length);
    }
    const b5 = fromObject(value);
    if (b5)
      return b5;
    if (typeof Symbol !== "undefined" && Symbol.toPrimitive != null && typeof value[Symbol.toPrimitive] === "function") {
      return Buffer3.from(value[Symbol.toPrimitive]("string"), encodingOrOffset, length);
    }
    throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value);
  }
  Buffer3.from = function(value, encodingOrOffset, length) {
    return from(value, encodingOrOffset, length);
  };
  Object.setPrototypeOf(Buffer3.prototype, Uint8Array.prototype);
  Object.setPrototypeOf(Buffer3, Uint8Array);
  function assertSize(size) {
    if (typeof size !== "number") {
      throw new TypeError('"size" argument must be of type number');
    } else if (size < 0) {
      throw new RangeError('The value "' + size + '" is invalid for option "size"');
    }
  }
  function alloc(size, fill, encoding) {
    assertSize(size);
    if (size <= 0) {
      return createBuffer(size);
    }
    if (fill !== void 0) {
      return typeof encoding === "string" ? createBuffer(size).fill(fill, encoding) : createBuffer(size).fill(fill);
    }
    return createBuffer(size);
  }
  Buffer3.alloc = function(size, fill, encoding) {
    return alloc(size, fill, encoding);
  };
  function allocUnsafe(size) {
    assertSize(size);
    return createBuffer(size < 0 ? 0 : checked(size) | 0);
  }
  Buffer3.allocUnsafe = function(size) {
    return allocUnsafe(size);
  };
  Buffer3.allocUnsafeSlow = function(size) {
    return allocUnsafe(size);
  };
  function fromString(string, encoding) {
    if (typeof encoding !== "string" || encoding === "") {
      encoding = "utf8";
    }
    if (!Buffer3.isEncoding(encoding)) {
      throw new TypeError("Unknown encoding: " + encoding);
    }
    const length = byteLength(string, encoding) | 0;
    let buf = createBuffer(length);
    const actual = buf.write(string, encoding);
    if (actual !== length) {
      buf = buf.slice(0, actual);
    }
    return buf;
  }
  function fromArrayLike(array) {
    const length = array.length < 0 ? 0 : checked(array.length) | 0;
    const buf = createBuffer(length);
    for (let i7 = 0; i7 < length; i7 += 1) {
      buf[i7] = array[i7] & 255;
    }
    return buf;
  }
  function fromArrayView(arrayView) {
    if (isInstance(arrayView, Uint8Array)) {
      const copy = new Uint8Array(arrayView);
      return fromArrayBuffer(copy.buffer, copy.byteOffset, copy.byteLength);
    }
    return fromArrayLike(arrayView);
  }
  function fromArrayBuffer(array, byteOffset, length) {
    if (byteOffset < 0 || array.byteLength < byteOffset) {
      throw new RangeError('"offset" is outside of buffer bounds');
    }
    if (array.byteLength < byteOffset + (length || 0)) {
      throw new RangeError('"length" is outside of buffer bounds');
    }
    let buf;
    if (byteOffset === void 0 && length === void 0) {
      buf = new Uint8Array(array);
    } else if (length === void 0) {
      buf = new Uint8Array(array, byteOffset);
    } else {
      buf = new Uint8Array(array, byteOffset, length);
    }
    Object.setPrototypeOf(buf, Buffer3.prototype);
    return buf;
  }
  function fromObject(obj) {
    if (Buffer3.isBuffer(obj)) {
      const len = checked(obj.length) | 0;
      const buf = createBuffer(len);
      if (buf.length === 0) {
        return buf;
      }
      obj.copy(buf, 0, 0, len);
      return buf;
    }
    if (obj.length !== void 0) {
      if (typeof obj.length !== "number" || numberIsNaN(obj.length)) {
        return createBuffer(0);
      }
      return fromArrayLike(obj);
    }
    if (obj.type === "Buffer" && Array.isArray(obj.data)) {
      return fromArrayLike(obj.data);
    }
  }
  function checked(length) {
    if (length >= K_MAX_LENGTH) {
      throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + K_MAX_LENGTH.toString(16) + " bytes");
    }
    return length | 0;
  }
  function SlowBuffer(length) {
    if (+length != length) {
      length = 0;
    }
    return Buffer3.alloc(+length);
  }
  Buffer3.isBuffer = function isBuffer4(b5) {
    return b5 != null && b5._isBuffer === true && b5 !== Buffer3.prototype;
  };
  Buffer3.compare = function compare(a7, b5) {
    if (isInstance(a7, Uint8Array))
      a7 = Buffer3.from(a7, a7.offset, a7.byteLength);
    if (isInstance(b5, Uint8Array))
      b5 = Buffer3.from(b5, b5.offset, b5.byteLength);
    if (!Buffer3.isBuffer(a7) || !Buffer3.isBuffer(b5)) {
      throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');
    }
    if (a7 === b5)
      return 0;
    let x4 = a7.length;
    let y6 = b5.length;
    for (let i7 = 0, len = Math.min(x4, y6); i7 < len; ++i7) {
      if (a7[i7] !== b5[i7]) {
        x4 = a7[i7];
        y6 = b5[i7];
        break;
      }
    }
    if (x4 < y6)
      return -1;
    if (y6 < x4)
      return 1;
    return 0;
  };
  Buffer3.isEncoding = function isEncoding(encoding) {
    switch (String(encoding).toLowerCase()) {
      case "hex":
      case "utf8":
      case "utf-8":
      case "ascii":
      case "latin1":
      case "binary":
      case "base64":
      case "ucs2":
      case "ucs-2":
      case "utf16le":
      case "utf-16le":
        return true;
      default:
        return false;
    }
  };
  Buffer3.concat = function concat(list, length) {
    if (!Array.isArray(list)) {
      throw new TypeError('"list" argument must be an Array of Buffers');
    }
    if (list.length === 0) {
      return Buffer3.alloc(0);
    }
    let i7;
    if (length === void 0) {
      length = 0;
      for (i7 = 0; i7 < list.length; ++i7) {
        length += list[i7].length;
      }
    }
    const buffer2 = Buffer3.allocUnsafe(length);
    let pos = 0;
    for (i7 = 0; i7 < list.length; ++i7) {
      let buf = list[i7];
      if (isInstance(buf, Uint8Array)) {
        if (pos + buf.length > buffer2.length) {
          if (!Buffer3.isBuffer(buf))
            buf = Buffer3.from(buf);
          buf.copy(buffer2, pos);
        } else {
          Uint8Array.prototype.set.call(buffer2, buf, pos);
        }
      } else if (!Buffer3.isBuffer(buf)) {
        throw new TypeError('"list" argument must be an Array of Buffers');
      } else {
        buf.copy(buffer2, pos);
      }
      pos += buf.length;
    }
    return buffer2;
  };
  function byteLength(string, encoding) {
    if (Buffer3.isBuffer(string)) {
      return string.length;
    }
    if (ArrayBuffer.isView(string) || isInstance(string, ArrayBuffer)) {
      return string.byteLength;
    }
    if (typeof string !== "string") {
      throw new TypeError('The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type ' + typeof string);
    }
    const len = string.length;
    const mustMatch = arguments.length > 2 && arguments[2] === true;
    if (!mustMatch && len === 0)
      return 0;
    let loweredCase = false;
    for (; ; ) {
      switch (encoding) {
        case "ascii":
        case "latin1":
        case "binary":
          return len;
        case "utf8":
        case "utf-8":
          return utf8ToBytes(string).length;
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return len * 2;
        case "hex":
          return len >>> 1;
        case "base64":
          return base64ToBytes(string).length;
        default:
          if (loweredCase) {
            return mustMatch ? -1 : utf8ToBytes(string).length;
          }
          encoding = ("" + encoding).toLowerCase();
          loweredCase = true;
      }
    }
  }
  Buffer3.byteLength = byteLength;
  function slowToString(encoding, start, end) {
    let loweredCase = false;
    if (start === void 0 || start < 0) {
      start = 0;
    }
    if (start > this.length) {
      return "";
    }
    if (end === void 0 || end > this.length) {
      end = this.length;
    }
    if (end <= 0) {
      return "";
    }
    end >>>= 0;
    start >>>= 0;
    if (end <= start) {
      return "";
    }
    if (!encoding)
      encoding = "utf8";
    while (true) {
      switch (encoding) {
        case "hex":
          return hexSlice(this, start, end);
        case "utf8":
        case "utf-8":
          return utf8Slice(this, start, end);
        case "ascii":
          return asciiSlice(this, start, end);
        case "latin1":
        case "binary":
          return latin1Slice(this, start, end);
        case "base64":
          return base64Slice(this, start, end);
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return utf16leSlice(this, start, end);
        default:
          if (loweredCase)
            throw new TypeError("Unknown encoding: " + encoding);
          encoding = (encoding + "").toLowerCase();
          loweredCase = true;
      }
    }
  }
  Buffer3.prototype._isBuffer = true;
  function swap(b5, n9, m6) {
    const i7 = b5[n9];
    b5[n9] = b5[m6];
    b5[m6] = i7;
  }
  Buffer3.prototype.swap16 = function swap16() {
    const len = this.length;
    if (len % 2 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 16-bits");
    }
    for (let i7 = 0; i7 < len; i7 += 2) {
      swap(this, i7, i7 + 1);
    }
    return this;
  };
  Buffer3.prototype.swap32 = function swap32() {
    const len = this.length;
    if (len % 4 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 32-bits");
    }
    for (let i7 = 0; i7 < len; i7 += 4) {
      swap(this, i7, i7 + 3);
      swap(this, i7 + 1, i7 + 2);
    }
    return this;
  };
  Buffer3.prototype.swap64 = function swap64() {
    const len = this.length;
    if (len % 8 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 64-bits");
    }
    for (let i7 = 0; i7 < len; i7 += 8) {
      swap(this, i7, i7 + 7);
      swap(this, i7 + 1, i7 + 6);
      swap(this, i7 + 2, i7 + 5);
      swap(this, i7 + 3, i7 + 4);
    }
    return this;
  };
  Buffer3.prototype.toString = function toString3() {
    const length = this.length;
    if (length === 0)
      return "";
    if (arguments.length === 0)
      return utf8Slice(this, 0, length);
    return slowToString.apply(this, arguments);
  };
  Buffer3.prototype.toLocaleString = Buffer3.prototype.toString;
  Buffer3.prototype.equals = function equals(b5) {
    if (!Buffer3.isBuffer(b5))
      throw new TypeError("Argument must be a Buffer");
    if (this === b5)
      return true;
    return Buffer3.compare(this, b5) === 0;
  };
  Buffer3.prototype.inspect = function inspect3() {
    let str = "";
    const max = exports$1.INSPECT_MAX_BYTES;
    str = this.toString("hex", 0, max).replace(/(.{2})/g, "$1 ").trim();
    if (this.length > max)
      str += " ... ";
    return "<Buffer " + str + ">";
  };
  if (customInspectSymbol) {
    Buffer3.prototype[customInspectSymbol] = Buffer3.prototype.inspect;
  }
  Buffer3.prototype.compare = function compare(target, start, end, thisStart, thisEnd) {
    if (isInstance(target, Uint8Array)) {
      target = Buffer3.from(target, target.offset, target.byteLength);
    }
    if (!Buffer3.isBuffer(target)) {
      throw new TypeError('The "target" argument must be one of type Buffer or Uint8Array. Received type ' + typeof target);
    }
    if (start === void 0) {
      start = 0;
    }
    if (end === void 0) {
      end = target ? target.length : 0;
    }
    if (thisStart === void 0) {
      thisStart = 0;
    }
    if (thisEnd === void 0) {
      thisEnd = this.length;
    }
    if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
      throw new RangeError("out of range index");
    }
    if (thisStart >= thisEnd && start >= end) {
      return 0;
    }
    if (thisStart >= thisEnd) {
      return -1;
    }
    if (start >= end) {
      return 1;
    }
    start >>>= 0;
    end >>>= 0;
    thisStart >>>= 0;
    thisEnd >>>= 0;
    if (this === target)
      return 0;
    let x4 = thisEnd - thisStart;
    let y6 = end - start;
    const len = Math.min(x4, y6);
    const thisCopy = this.slice(thisStart, thisEnd);
    const targetCopy = target.slice(start, end);
    for (let i7 = 0; i7 < len; ++i7) {
      if (thisCopy[i7] !== targetCopy[i7]) {
        x4 = thisCopy[i7];
        y6 = targetCopy[i7];
        break;
      }
    }
    if (x4 < y6)
      return -1;
    if (y6 < x4)
      return 1;
    return 0;
  };
  function bidirectionalIndexOf(buffer2, val, byteOffset, encoding, dir) {
    if (buffer2.length === 0)
      return -1;
    if (typeof byteOffset === "string") {
      encoding = byteOffset;
      byteOffset = 0;
    } else if (byteOffset > 2147483647) {
      byteOffset = 2147483647;
    } else if (byteOffset < -2147483648) {
      byteOffset = -2147483648;
    }
    byteOffset = +byteOffset;
    if (numberIsNaN(byteOffset)) {
      byteOffset = dir ? 0 : buffer2.length - 1;
    }
    if (byteOffset < 0)
      byteOffset = buffer2.length + byteOffset;
    if (byteOffset >= buffer2.length) {
      if (dir)
        return -1;
      else
        byteOffset = buffer2.length - 1;
    } else if (byteOffset < 0) {
      if (dir)
        byteOffset = 0;
      else
        return -1;
    }
    if (typeof val === "string") {
      val = Buffer3.from(val, encoding);
    }
    if (Buffer3.isBuffer(val)) {
      if (val.length === 0) {
        return -1;
      }
      return arrayIndexOf(buffer2, val, byteOffset, encoding, dir);
    } else if (typeof val === "number") {
      val = val & 255;
      if (typeof Uint8Array.prototype.indexOf === "function") {
        if (dir) {
          return Uint8Array.prototype.indexOf.call(buffer2, val, byteOffset);
        } else {
          return Uint8Array.prototype.lastIndexOf.call(buffer2, val, byteOffset);
        }
      }
      return arrayIndexOf(buffer2, [val], byteOffset, encoding, dir);
    }
    throw new TypeError("val must be string, number or Buffer");
  }
  function arrayIndexOf(arr, val, byteOffset, encoding, dir) {
    let indexSize = 1;
    let arrLength = arr.length;
    let valLength = val.length;
    if (encoding !== void 0) {
      encoding = String(encoding).toLowerCase();
      if (encoding === "ucs2" || encoding === "ucs-2" || encoding === "utf16le" || encoding === "utf-16le") {
        if (arr.length < 2 || val.length < 2) {
          return -1;
        }
        indexSize = 2;
        arrLength /= 2;
        valLength /= 2;
        byteOffset /= 2;
      }
    }
    function read2(buf, i8) {
      if (indexSize === 1) {
        return buf[i8];
      } else {
        return buf.readUInt16BE(i8 * indexSize);
      }
    }
    let i7;
    if (dir) {
      let foundIndex = -1;
      for (i7 = byteOffset; i7 < arrLength; i7++) {
        if (read2(arr, i7) === read2(val, foundIndex === -1 ? 0 : i7 - foundIndex)) {
          if (foundIndex === -1)
            foundIndex = i7;
          if (i7 - foundIndex + 1 === valLength)
            return foundIndex * indexSize;
        } else {
          if (foundIndex !== -1)
            i7 -= i7 - foundIndex;
          foundIndex = -1;
        }
      }
    } else {
      if (byteOffset + valLength > arrLength)
        byteOffset = arrLength - valLength;
      for (i7 = byteOffset; i7 >= 0; i7--) {
        let found = true;
        for (let j4 = 0; j4 < valLength; j4++) {
          if (read2(arr, i7 + j4) !== read2(val, j4)) {
            found = false;
            break;
          }
        }
        if (found)
          return i7;
      }
    }
    return -1;
  }
  Buffer3.prototype.includes = function includes(val, byteOffset, encoding) {
    return this.indexOf(val, byteOffset, encoding) !== -1;
  };
  Buffer3.prototype.indexOf = function indexOf(val, byteOffset, encoding) {
    return bidirectionalIndexOf(this, val, byteOffset, encoding, true);
  };
  Buffer3.prototype.lastIndexOf = function lastIndexOf(val, byteOffset, encoding) {
    return bidirectionalIndexOf(this, val, byteOffset, encoding, false);
  };
  function hexWrite(buf, string, offset, length) {
    offset = Number(offset) || 0;
    const remaining = buf.length - offset;
    if (!length) {
      length = remaining;
    } else {
      length = Number(length);
      if (length > remaining) {
        length = remaining;
      }
    }
    const strLen = string.length;
    if (length > strLen / 2) {
      length = strLen / 2;
    }
    let i7;
    for (i7 = 0; i7 < length; ++i7) {
      const parsed = parseInt(string.substr(i7 * 2, 2), 16);
      if (numberIsNaN(parsed))
        return i7;
      buf[offset + i7] = parsed;
    }
    return i7;
  }
  function utf8Write(buf, string, offset, length) {
    return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length);
  }
  function asciiWrite(buf, string, offset, length) {
    return blitBuffer(asciiToBytes(string), buf, offset, length);
  }
  function base64Write(buf, string, offset, length) {
    return blitBuffer(base64ToBytes(string), buf, offset, length);
  }
  function ucs2Write(buf, string, offset, length) {
    return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length);
  }
  Buffer3.prototype.write = function write2(string, offset, length, encoding) {
    if (offset === void 0) {
      encoding = "utf8";
      length = this.length;
      offset = 0;
    } else if (length === void 0 && typeof offset === "string") {
      encoding = offset;
      length = this.length;
      offset = 0;
    } else if (isFinite(offset)) {
      offset = offset >>> 0;
      if (isFinite(length)) {
        length = length >>> 0;
        if (encoding === void 0)
          encoding = "utf8";
      } else {
        encoding = length;
        length = void 0;
      }
    } else {
      throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");
    }
    const remaining = this.length - offset;
    if (length === void 0 || length > remaining)
      length = remaining;
    if (string.length > 0 && (length < 0 || offset < 0) || offset > this.length) {
      throw new RangeError("Attempt to write outside buffer bounds");
    }
    if (!encoding)
      encoding = "utf8";
    let loweredCase = false;
    for (; ; ) {
      switch (encoding) {
        case "hex":
          return hexWrite(this, string, offset, length);
        case "utf8":
        case "utf-8":
          return utf8Write(this, string, offset, length);
        case "ascii":
        case "latin1":
        case "binary":
          return asciiWrite(this, string, offset, length);
        case "base64":
          return base64Write(this, string, offset, length);
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return ucs2Write(this, string, offset, length);
        default:
          if (loweredCase)
            throw new TypeError("Unknown encoding: " + encoding);
          encoding = ("" + encoding).toLowerCase();
          loweredCase = true;
      }
    }
  };
  Buffer3.prototype.toJSON = function toJSON2() {
    return {
      type: "Buffer",
      data: Array.prototype.slice.call(this._arr || this, 0)
    };
  };
  function base64Slice(buf, start, end) {
    if (start === 0 && end === buf.length) {
      return base642.fromByteArray(buf);
    } else {
      return base642.fromByteArray(buf.slice(start, end));
    }
  }
  function utf8Slice(buf, start, end) {
    end = Math.min(buf.length, end);
    const res = [];
    let i7 = start;
    while (i7 < end) {
      const firstByte = buf[i7];
      let codePoint = null;
      let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
      if (i7 + bytesPerSequence <= end) {
        let secondByte, thirdByte, fourthByte, tempCodePoint;
        switch (bytesPerSequence) {
          case 1:
            if (firstByte < 128) {
              codePoint = firstByte;
            }
            break;
          case 2:
            secondByte = buf[i7 + 1];
            if ((secondByte & 192) === 128) {
              tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
              if (tempCodePoint > 127) {
                codePoint = tempCodePoint;
              }
            }
            break;
          case 3:
            secondByte = buf[i7 + 1];
            thirdByte = buf[i7 + 2];
            if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
              tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
              if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                codePoint = tempCodePoint;
              }
            }
            break;
          case 4:
            secondByte = buf[i7 + 1];
            thirdByte = buf[i7 + 2];
            fourthByte = buf[i7 + 3];
            if ((secondByte & 192) === 128 && (thirdByte & 192) === 128 && (fourthByte & 192) === 128) {
              tempCodePoint = (firstByte & 15) << 18 | (secondByte & 63) << 12 | (thirdByte & 63) << 6 | fourthByte & 63;
              if (tempCodePoint > 65535 && tempCodePoint < 1114112) {
                codePoint = tempCodePoint;
              }
            }
        }
      }
      if (codePoint === null) {
        codePoint = 65533;
        bytesPerSequence = 1;
      } else if (codePoint > 65535) {
        codePoint -= 65536;
        res.push(codePoint >>> 10 & 1023 | 55296);
        codePoint = 56320 | codePoint & 1023;
      }
      res.push(codePoint);
      i7 += bytesPerSequence;
    }
    return decodeCodePointsArray(res);
  }
  const MAX_ARGUMENTS_LENGTH = 4096;
  function decodeCodePointsArray(codePoints) {
    const len = codePoints.length;
    if (len <= MAX_ARGUMENTS_LENGTH) {
      return String.fromCharCode.apply(String, codePoints);
    }
    let res = "";
    let i7 = 0;
    while (i7 < len) {
      res += String.fromCharCode.apply(String, codePoints.slice(i7, i7 += MAX_ARGUMENTS_LENGTH));
    }
    return res;
  }
  function asciiSlice(buf, start, end) {
    let ret = "";
    end = Math.min(buf.length, end);
    for (let i7 = start; i7 < end; ++i7) {
      ret += String.fromCharCode(buf[i7] & 127);
    }
    return ret;
  }
  function latin1Slice(buf, start, end) {
    let ret = "";
    end = Math.min(buf.length, end);
    for (let i7 = start; i7 < end; ++i7) {
      ret += String.fromCharCode(buf[i7]);
    }
    return ret;
  }
  function hexSlice(buf, start, end) {
    const len = buf.length;
    if (!start || start < 0)
      start = 0;
    if (!end || end < 0 || end > len)
      end = len;
    let out = "";
    for (let i7 = start; i7 < end; ++i7) {
      out += hexSliceLookupTable[buf[i7]];
    }
    return out;
  }
  function utf16leSlice(buf, start, end) {
    const bytes = buf.slice(start, end);
    let res = "";
    for (let i7 = 0; i7 < bytes.length - 1; i7 += 2) {
      res += String.fromCharCode(bytes[i7] + bytes[i7 + 1] * 256);
    }
    return res;
  }
  Buffer3.prototype.slice = function slice(start, end) {
    const len = this.length;
    start = ~~start;
    end = end === void 0 ? len : ~~end;
    if (start < 0) {
      start += len;
      if (start < 0)
        start = 0;
    } else if (start > len) {
      start = len;
    }
    if (end < 0) {
      end += len;
      if (end < 0)
        end = 0;
    } else if (end > len) {
      end = len;
    }
    if (end < start)
      end = start;
    const newBuf = this.subarray(start, end);
    Object.setPrototypeOf(newBuf, Buffer3.prototype);
    return newBuf;
  };
  function checkOffset(offset, ext, length) {
    if (offset % 1 !== 0 || offset < 0)
      throw new RangeError("offset is not uint");
    if (offset + ext > length)
      throw new RangeError("Trying to access beyond buffer length");
  }
  Buffer3.prototype.readUintLE = Buffer3.prototype.readUIntLE = function readUIntLE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let val = this[offset];
    let mul = 1;
    let i7 = 0;
    while (++i7 < byteLength2 && (mul *= 256)) {
      val += this[offset + i7] * mul;
    }
    return val;
  };
  Buffer3.prototype.readUintBE = Buffer3.prototype.readUIntBE = function readUIntBE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      checkOffset(offset, byteLength2, this.length);
    }
    let val = this[offset + --byteLength2];
    let mul = 1;
    while (byteLength2 > 0 && (mul *= 256)) {
      val += this[offset + --byteLength2] * mul;
    }
    return val;
  };
  Buffer3.prototype.readUint8 = Buffer3.prototype.readUInt8 = function readUInt8(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 1, this.length);
    return this[offset];
  };
  Buffer3.prototype.readUint16LE = Buffer3.prototype.readUInt16LE = function readUInt16LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    return this[offset] | this[offset + 1] << 8;
  };
  Buffer3.prototype.readUint16BE = Buffer3.prototype.readUInt16BE = function readUInt16BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    return this[offset] << 8 | this[offset + 1];
  };
  Buffer3.prototype.readUint32LE = Buffer3.prototype.readUInt32LE = function readUInt32LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return (this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16) + this[offset + 3] * 16777216;
  };
  Buffer3.prototype.readUint32BE = Buffer3.prototype.readUInt32BE = function readUInt32BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] * 16777216 + (this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3]);
  };
  Buffer3.prototype.readBigUInt64LE = defineBigIntMethod(function readBigUInt64LE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const lo = first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24;
    const hi = this[++offset] + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + last * 2 ** 24;
    return BigInt(lo) + (BigInt(hi) << BigInt(32));
  });
  Buffer3.prototype.readBigUInt64BE = defineBigIntMethod(function readBigUInt64BE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const hi = first * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
    const lo = this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last;
    return (BigInt(hi) << BigInt(32)) + BigInt(lo);
  });
  Buffer3.prototype.readIntLE = function readIntLE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let val = this[offset];
    let mul = 1;
    let i7 = 0;
    while (++i7 < byteLength2 && (mul *= 256)) {
      val += this[offset + i7] * mul;
    }
    mul *= 128;
    if (val >= mul)
      val -= Math.pow(2, 8 * byteLength2);
    return val;
  };
  Buffer3.prototype.readIntBE = function readIntBE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let i7 = byteLength2;
    let mul = 1;
    let val = this[offset + --i7];
    while (i7 > 0 && (mul *= 256)) {
      val += this[offset + --i7] * mul;
    }
    mul *= 128;
    if (val >= mul)
      val -= Math.pow(2, 8 * byteLength2);
    return val;
  };
  Buffer3.prototype.readInt8 = function readInt8(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 1, this.length);
    if (!(this[offset] & 128))
      return this[offset];
    return (255 - this[offset] + 1) * -1;
  };
  Buffer3.prototype.readInt16LE = function readInt16LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    const val = this[offset] | this[offset + 1] << 8;
    return val & 32768 ? val | 4294901760 : val;
  };
  Buffer3.prototype.readInt16BE = function readInt16BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    const val = this[offset + 1] | this[offset] << 8;
    return val & 32768 ? val | 4294901760 : val;
  };
  Buffer3.prototype.readInt32LE = function readInt32LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16 | this[offset + 3] << 24;
  };
  Buffer3.prototype.readInt32BE = function readInt32BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] << 24 | this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3];
  };
  Buffer3.prototype.readBigInt64LE = defineBigIntMethod(function readBigInt64LE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const val = this[offset + 4] + this[offset + 5] * 2 ** 8 + this[offset + 6] * 2 ** 16 + (last << 24);
    return (BigInt(val) << BigInt(32)) + BigInt(first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24);
  });
  Buffer3.prototype.readBigInt64BE = defineBigIntMethod(function readBigInt64BE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const val = (first << 24) + // Overflow
    this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
    return (BigInt(val) << BigInt(32)) + BigInt(this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last);
  });
  Buffer3.prototype.readFloatLE = function readFloatLE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return ieee754.read(this, offset, true, 23, 4);
  };
  Buffer3.prototype.readFloatBE = function readFloatBE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return ieee754.read(this, offset, false, 23, 4);
  };
  Buffer3.prototype.readDoubleLE = function readDoubleLE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 8, this.length);
    return ieee754.read(this, offset, true, 52, 8);
  };
  Buffer3.prototype.readDoubleBE = function readDoubleBE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 8, this.length);
    return ieee754.read(this, offset, false, 52, 8);
  };
  function checkInt(buf, value, offset, ext, max, min) {
    if (!Buffer3.isBuffer(buf))
      throw new TypeError('"buffer" argument must be a Buffer instance');
    if (value > max || value < min)
      throw new RangeError('"value" argument is out of bounds');
    if (offset + ext > buf.length)
      throw new RangeError("Index out of range");
  }
  Buffer3.prototype.writeUintLE = Buffer3.prototype.writeUIntLE = function writeUIntLE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
      checkInt(this, value, offset, byteLength2, maxBytes, 0);
    }
    let mul = 1;
    let i7 = 0;
    this[offset] = value & 255;
    while (++i7 < byteLength2 && (mul *= 256)) {
      this[offset + i7] = value / mul & 255;
    }
    return offset + byteLength2;
  };
  Buffer3.prototype.writeUintBE = Buffer3.prototype.writeUIntBE = function writeUIntBE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
      checkInt(this, value, offset, byteLength2, maxBytes, 0);
    }
    let i7 = byteLength2 - 1;
    let mul = 1;
    this[offset + i7] = value & 255;
    while (--i7 >= 0 && (mul *= 256)) {
      this[offset + i7] = value / mul & 255;
    }
    return offset + byteLength2;
  };
  Buffer3.prototype.writeUint8 = Buffer3.prototype.writeUInt8 = function writeUInt8(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 1, 255, 0);
    this[offset] = value & 255;
    return offset + 1;
  };
  Buffer3.prototype.writeUint16LE = Buffer3.prototype.writeUInt16LE = function writeUInt16LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 65535, 0);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    return offset + 2;
  };
  Buffer3.prototype.writeUint16BE = Buffer3.prototype.writeUInt16BE = function writeUInt16BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 65535, 0);
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
    return offset + 2;
  };
  Buffer3.prototype.writeUint32LE = Buffer3.prototype.writeUInt32LE = function writeUInt32LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 4294967295, 0);
    this[offset + 3] = value >>> 24;
    this[offset + 2] = value >>> 16;
    this[offset + 1] = value >>> 8;
    this[offset] = value & 255;
    return offset + 4;
  };
  Buffer3.prototype.writeUint32BE = Buffer3.prototype.writeUInt32BE = function writeUInt32BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 4294967295, 0);
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
    return offset + 4;
  };
  function wrtBigUInt64LE(buf, value, offset, min, max) {
    checkIntBI(value, min, max, buf, offset, 7);
    let lo = Number(value & BigInt(4294967295));
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    let hi = Number(value >> BigInt(32) & BigInt(4294967295));
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    return offset;
  }
  function wrtBigUInt64BE(buf, value, offset, min, max) {
    checkIntBI(value, min, max, buf, offset, 7);
    let lo = Number(value & BigInt(4294967295));
    buf[offset + 7] = lo;
    lo = lo >> 8;
    buf[offset + 6] = lo;
    lo = lo >> 8;
    buf[offset + 5] = lo;
    lo = lo >> 8;
    buf[offset + 4] = lo;
    let hi = Number(value >> BigInt(32) & BigInt(4294967295));
    buf[offset + 3] = hi;
    hi = hi >> 8;
    buf[offset + 2] = hi;
    hi = hi >> 8;
    buf[offset + 1] = hi;
    hi = hi >> 8;
    buf[offset] = hi;
    return offset + 8;
  }
  Buffer3.prototype.writeBigUInt64LE = defineBigIntMethod(function writeBigUInt64LE(value, offset = 0) {
    return wrtBigUInt64LE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
  });
  Buffer3.prototype.writeBigUInt64BE = defineBigIntMethod(function writeBigUInt64BE(value, offset = 0) {
    return wrtBigUInt64BE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
  });
  Buffer3.prototype.writeIntLE = function writeIntLE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      const limit = Math.pow(2, 8 * byteLength2 - 1);
      checkInt(this, value, offset, byteLength2, limit - 1, -limit);
    }
    let i7 = 0;
    let mul = 1;
    let sub = 0;
    this[offset] = value & 255;
    while (++i7 < byteLength2 && (mul *= 256)) {
      if (value < 0 && sub === 0 && this[offset + i7 - 1] !== 0) {
        sub = 1;
      }
      this[offset + i7] = (value / mul >> 0) - sub & 255;
    }
    return offset + byteLength2;
  };
  Buffer3.prototype.writeIntBE = function writeIntBE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      const limit = Math.pow(2, 8 * byteLength2 - 1);
      checkInt(this, value, offset, byteLength2, limit - 1, -limit);
    }
    let i7 = byteLength2 - 1;
    let mul = 1;
    let sub = 0;
    this[offset + i7] = value & 255;
    while (--i7 >= 0 && (mul *= 256)) {
      if (value < 0 && sub === 0 && this[offset + i7 + 1] !== 0) {
        sub = 1;
      }
      this[offset + i7] = (value / mul >> 0) - sub & 255;
    }
    return offset + byteLength2;
  };
  Buffer3.prototype.writeInt8 = function writeInt8(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 1, 127, -128);
    if (value < 0)
      value = 255 + value + 1;
    this[offset] = value & 255;
    return offset + 1;
  };
  Buffer3.prototype.writeInt16LE = function writeInt16LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 32767, -32768);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    return offset + 2;
  };
  Buffer3.prototype.writeInt16BE = function writeInt16BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 32767, -32768);
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
    return offset + 2;
  };
  Buffer3.prototype.writeInt32LE = function writeInt32LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 2147483647, -2147483648);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    this[offset + 2] = value >>> 16;
    this[offset + 3] = value >>> 24;
    return offset + 4;
  };
  Buffer3.prototype.writeInt32BE = function writeInt32BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 2147483647, -2147483648);
    if (value < 0)
      value = 4294967295 + value + 1;
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
    return offset + 4;
  };
  Buffer3.prototype.writeBigInt64LE = defineBigIntMethod(function writeBigInt64LE(value, offset = 0) {
    return wrtBigUInt64LE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
  });
  Buffer3.prototype.writeBigInt64BE = defineBigIntMethod(function writeBigInt64BE(value, offset = 0) {
    return wrtBigUInt64BE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
  });
  function checkIEEE754(buf, value, offset, ext, max, min) {
    if (offset + ext > buf.length)
      throw new RangeError("Index out of range");
    if (offset < 0)
      throw new RangeError("Index out of range");
  }
  function writeFloat(buf, value, offset, littleEndian, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      checkIEEE754(buf, value, offset, 4);
    }
    ieee754.write(buf, value, offset, littleEndian, 23, 4);
    return offset + 4;
  }
  Buffer3.prototype.writeFloatLE = function writeFloatLE(value, offset, noAssert) {
    return writeFloat(this, value, offset, true, noAssert);
  };
  Buffer3.prototype.writeFloatBE = function writeFloatBE(value, offset, noAssert) {
    return writeFloat(this, value, offset, false, noAssert);
  };
  function writeDouble(buf, value, offset, littleEndian, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      checkIEEE754(buf, value, offset, 8);
    }
    ieee754.write(buf, value, offset, littleEndian, 52, 8);
    return offset + 8;
  }
  Buffer3.prototype.writeDoubleLE = function writeDoubleLE(value, offset, noAssert) {
    return writeDouble(this, value, offset, true, noAssert);
  };
  Buffer3.prototype.writeDoubleBE = function writeDoubleBE(value, offset, noAssert) {
    return writeDouble(this, value, offset, false, noAssert);
  };
  Buffer3.prototype.copy = function copy(target, targetStart, start, end) {
    if (!Buffer3.isBuffer(target))
      throw new TypeError("argument should be a Buffer");
    if (!start)
      start = 0;
    if (!end && end !== 0)
      end = this.length;
    if (targetStart >= target.length)
      targetStart = target.length;
    if (!targetStart)
      targetStart = 0;
    if (end > 0 && end < start)
      end = start;
    if (end === start)
      return 0;
    if (target.length === 0 || this.length === 0)
      return 0;
    if (targetStart < 0) {
      throw new RangeError("targetStart out of bounds");
    }
    if (start < 0 || start >= this.length)
      throw new RangeError("Index out of range");
    if (end < 0)
      throw new RangeError("sourceEnd out of bounds");
    if (end > this.length)
      end = this.length;
    if (target.length - targetStart < end - start) {
      end = target.length - targetStart + start;
    }
    const len = end - start;
    if (this === target && typeof Uint8Array.prototype.copyWithin === "function") {
      this.copyWithin(targetStart, start, end);
    } else {
      Uint8Array.prototype.set.call(target, this.subarray(start, end), targetStart);
    }
    return len;
  };
  Buffer3.prototype.fill = function fill(val, start, end, encoding) {
    if (typeof val === "string") {
      if (typeof start === "string") {
        encoding = start;
        start = 0;
        end = this.length;
      } else if (typeof end === "string") {
        encoding = end;
        end = this.length;
      }
      if (encoding !== void 0 && typeof encoding !== "string") {
        throw new TypeError("encoding must be a string");
      }
      if (typeof encoding === "string" && !Buffer3.isEncoding(encoding)) {
        throw new TypeError("Unknown encoding: " + encoding);
      }
      if (val.length === 1) {
        const code = val.charCodeAt(0);
        if (encoding === "utf8" && code < 128 || encoding === "latin1") {
          val = code;
        }
      }
    } else if (typeof val === "number") {
      val = val & 255;
    } else if (typeof val === "boolean") {
      val = Number(val);
    }
    if (start < 0 || this.length < start || this.length < end) {
      throw new RangeError("Out of range index");
    }
    if (end <= start) {
      return this;
    }
    start = start >>> 0;
    end = end === void 0 ? this.length : end >>> 0;
    if (!val)
      val = 0;
    let i7;
    if (typeof val === "number") {
      for (i7 = start; i7 < end; ++i7) {
        this[i7] = val;
      }
    } else {
      const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
      const len = bytes.length;
      if (len === 0) {
        throw new TypeError('The value "' + val + '" is invalid for argument "value"');
      }
      for (i7 = 0; i7 < end - start; ++i7) {
        this[i7 + start] = bytes[i7 % len];
      }
    }
    return this;
  };
  const errors = {};
  function E4(sym, getMessage, Base) {
    errors[sym] = class NodeError extends Base {
      constructor() {
        super();
        Object.defineProperty(this, "message", {
          value: getMessage.apply(this, arguments),
          writable: true,
          configurable: true
        });
        this.name = `${this.name} [${sym}]`;
        this.stack;
        delete this.name;
      }
      get code() {
        return sym;
      }
      set code(value) {
        Object.defineProperty(this, "code", {
          configurable: true,
          enumerable: true,
          value,
          writable: true
        });
      }
      toString() {
        return `${this.name} [${sym}]: ${this.message}`;
      }
    };
  }
  E4("ERR_BUFFER_OUT_OF_BOUNDS", function(name3) {
    if (name3) {
      return `${name3} is outside of buffer bounds`;
    }
    return "Attempt to access memory outside buffer bounds";
  }, RangeError);
  E4("ERR_INVALID_ARG_TYPE", function(name3, actual) {
    return `The "${name3}" argument must be of type number. Received type ${typeof actual}`;
  }, TypeError);
  E4("ERR_OUT_OF_RANGE", function(str, range, input) {
    let msg = `The value of "${str}" is out of range.`;
    let received = input;
    if (Number.isInteger(input) && Math.abs(input) > 2 ** 32) {
      received = addNumericalSeparator(String(input));
    } else if (typeof input === "bigint") {
      received = String(input);
      if (input > BigInt(2) ** BigInt(32) || input < -(BigInt(2) ** BigInt(32))) {
        received = addNumericalSeparator(received);
      }
      received += "n";
    }
    msg += ` It must be ${range}. Received ${received}`;
    return msg;
  }, RangeError);
  function addNumericalSeparator(val) {
    let res = "";
    let i7 = val.length;
    const start = val[0] === "-" ? 1 : 0;
    for (; i7 >= start + 4; i7 -= 3) {
      res = `_${val.slice(i7 - 3, i7)}${res}`;
    }
    return `${val.slice(0, i7)}${res}`;
  }
  function checkBounds(buf, offset, byteLength2) {
    validateNumber(offset, "offset");
    if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
      boundsError(offset, buf.length - (byteLength2 + 1));
    }
  }
  function checkIntBI(value, min, max, buf, offset, byteLength2) {
    if (value > max || value < min) {
      const n9 = typeof min === "bigint" ? "n" : "";
      let range;
      if (byteLength2 > 3) {
        if (min === 0 || min === BigInt(0)) {
          range = `>= 0${n9} and < 2${n9} ** ${(byteLength2 + 1) * 8}${n9}`;
        } else {
          range = `>= -(2${n9} ** ${(byteLength2 + 1) * 8 - 1}${n9}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n9}`;
        }
      } else {
        range = `>= ${min}${n9} and <= ${max}${n9}`;
      }
      throw new errors.ERR_OUT_OF_RANGE("value", range, value);
    }
    checkBounds(buf, offset, byteLength2);
  }
  function validateNumber(value, name3) {
    if (typeof value !== "number") {
      throw new errors.ERR_INVALID_ARG_TYPE(name3, "number", value);
    }
  }
  function boundsError(value, length, type2) {
    if (Math.floor(value) !== value) {
      validateNumber(value, type2);
      throw new errors.ERR_OUT_OF_RANGE(type2 || "offset", "an integer", value);
    }
    if (length < 0) {
      throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
    }
    throw new errors.ERR_OUT_OF_RANGE(type2 || "offset", `>= ${type2 ? 1 : 0} and <= ${length}`, value);
  }
  const INVALID_BASE64_RE = /[^+/0-9A-Za-z-_]/g;
  function base64clean(str) {
    str = str.split("=")[0];
    str = str.trim().replace(INVALID_BASE64_RE, "");
    if (str.length < 2)
      return "";
    while (str.length % 4 !== 0) {
      str = str + "=";
    }
    return str;
  }
  function utf8ToBytes(string, units) {
    units = units || Infinity;
    let codePoint;
    const length = string.length;
    let leadSurrogate = null;
    const bytes = [];
    for (let i7 = 0; i7 < length; ++i7) {
      codePoint = string.charCodeAt(i7);
      if (codePoint > 55295 && codePoint < 57344) {
        if (!leadSurrogate) {
          if (codePoint > 56319) {
            if ((units -= 3) > -1)
              bytes.push(239, 191, 189);
            continue;
          } else if (i7 + 1 === length) {
            if ((units -= 3) > -1)
              bytes.push(239, 191, 189);
            continue;
          }
          leadSurrogate = codePoint;
          continue;
        }
        if (codePoint < 56320) {
          if ((units -= 3) > -1)
            bytes.push(239, 191, 189);
          leadSurrogate = codePoint;
          continue;
        }
        codePoint = (leadSurrogate - 55296 << 10 | codePoint - 56320) + 65536;
      } else if (leadSurrogate) {
        if ((units -= 3) > -1)
          bytes.push(239, 191, 189);
      }
      leadSurrogate = null;
      if (codePoint < 128) {
        if ((units -= 1) < 0)
          break;
        bytes.push(codePoint);
      } else if (codePoint < 2048) {
        if ((units -= 2) < 0)
          break;
        bytes.push(codePoint >> 6 | 192, codePoint & 63 | 128);
      } else if (codePoint < 65536) {
        if ((units -= 3) < 0)
          break;
        bytes.push(codePoint >> 12 | 224, codePoint >> 6 & 63 | 128, codePoint & 63 | 128);
      } else if (codePoint < 1114112) {
        if ((units -= 4) < 0)
          break;
        bytes.push(codePoint >> 18 | 240, codePoint >> 12 & 63 | 128, codePoint >> 6 & 63 | 128, codePoint & 63 | 128);
      } else {
        throw new Error("Invalid code point");
      }
    }
    return bytes;
  }
  function asciiToBytes(str) {
    const byteArray = [];
    for (let i7 = 0; i7 < str.length; ++i7) {
      byteArray.push(str.charCodeAt(i7) & 255);
    }
    return byteArray;
  }
  function utf16leToBytes(str, units) {
    let c7, hi, lo;
    const byteArray = [];
    for (let i7 = 0; i7 < str.length; ++i7) {
      if ((units -= 2) < 0)
        break;
      c7 = str.charCodeAt(i7);
      hi = c7 >> 8;
      lo = c7 % 256;
      byteArray.push(lo);
      byteArray.push(hi);
    }
    return byteArray;
  }
  function base64ToBytes(str) {
    return base642.toByteArray(base64clean(str));
  }
  function blitBuffer(src, dst, offset, length) {
    let i7;
    for (i7 = 0; i7 < length; ++i7) {
      if (i7 + offset >= dst.length || i7 >= src.length)
        break;
      dst[i7 + offset] = src[i7];
    }
    return i7;
  }
  function isInstance(obj, type2) {
    return obj instanceof type2 || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type2.name;
  }
  function numberIsNaN(obj) {
    return obj !== obj;
  }
  const hexSliceLookupTable = function() {
    const alphabet = "0123456789abcdef";
    const table = new Array(256);
    for (let i7 = 0; i7 < 16; ++i7) {
      const i16 = i7 * 16;
      for (let j4 = 0; j4 < 16; ++j4) {
        table[i16 + j4] = alphabet[i7] + alphabet[j4];
      }
    }
    return table;
  }();
  function defineBigIntMethod(fn) {
    return typeof BigInt === "undefined" ? BufferBigIntNotDefined : fn;
  }
  function BufferBigIntNotDefined() {
    throw new Error("BigInt not supported");
  }
  return exports$1;
}
var exports$3, _dewExec$2, exports$2, _dewExec$1, exports$1, _dewExec, exports, Buffer2;
var init_buffer = __esm({
  "node_modules/@jspm/core/nodelibs/browser/buffer.js"() {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    exports$3 = {};
    _dewExec$2 = false;
    exports$2 = {};
    _dewExec$1 = false;
    exports$1 = {};
    _dewExec = false;
    exports = dew();
    exports["Buffer"];
    exports["SlowBuffer"];
    exports["INSPECT_MAX_BYTES"];
    exports["kMaxLength"];
    Buffer2 = exports.Buffer;
    exports.INSPECT_MAX_BYTES;
    exports.kMaxLength;
  }
});

// node_modules/esbuild-plugin-polyfill-node/polyfills/buffer.js
var init_buffer2 = __esm({
  "node_modules/esbuild-plugin-polyfill-node/polyfills/buffer.js"() {
    init_buffer();
  }
});

// node_modules/@jspm/core/nodelibs/browser/chunk-4bd36a8f.js
function o() {
  o.init.call(this);
}
function u(e10) {
  if ("function" != typeof e10)
    throw new TypeError('The "listener" argument must be of type Function. Received type ' + typeof e10);
}
function f(e10) {
  return void 0 === e10._maxListeners ? o.defaultMaxListeners : e10._maxListeners;
}
function v(e10, t9, n9, r10) {
  var i7, o9, s6, v7;
  if (u(n9), void 0 === (o9 = e10._events) ? (o9 = e10._events = /* @__PURE__ */ Object.create(null), e10._eventsCount = 0) : (void 0 !== o9.newListener && (e10.emit("newListener", t9, n9.listener ? n9.listener : n9), o9 = e10._events), s6 = o9[t9]), void 0 === s6)
    s6 = o9[t9] = n9, ++e10._eventsCount;
  else if ("function" == typeof s6 ? s6 = o9[t9] = r10 ? [n9, s6] : [s6, n9] : r10 ? s6.unshift(n9) : s6.push(n9), (i7 = f(e10)) > 0 && s6.length > i7 && !s6.warned) {
    s6.warned = true;
    var a7 = new Error("Possible EventEmitter memory leak detected. " + s6.length + " " + String(t9) + " listeners added. Use emitter.setMaxListeners() to increase limit");
    a7.name = "MaxListenersExceededWarning", a7.emitter = e10, a7.type = t9, a7.count = s6.length, v7 = a7, console && console.warn && console.warn(v7);
  }
  return e10;
}
function a() {
  if (!this.fired)
    return this.target.removeListener(this.type, this.wrapFn), this.fired = true, 0 === arguments.length ? this.listener.call(this.target) : this.listener.apply(this.target, arguments);
}
function l(e10, t9, n9) {
  var r10 = { fired: false, wrapFn: void 0, target: e10, type: t9, listener: n9 }, i7 = a.bind(r10);
  return i7.listener = n9, r10.wrapFn = i7, i7;
}
function h(e10, t9, n9) {
  var r10 = e10._events;
  if (void 0 === r10)
    return [];
  var i7 = r10[t9];
  return void 0 === i7 ? [] : "function" == typeof i7 ? n9 ? [i7.listener || i7] : [i7] : n9 ? function(e11) {
    for (var t10 = new Array(e11.length), n10 = 0; n10 < t10.length; ++n10)
      t10[n10] = e11[n10].listener || e11[n10];
    return t10;
  }(i7) : c(i7, i7.length);
}
function p(e10) {
  var t9 = this._events;
  if (void 0 !== t9) {
    var n9 = t9[e10];
    if ("function" == typeof n9)
      return 1;
    if (void 0 !== n9)
      return n9.length;
  }
  return 0;
}
function c(e10, t9) {
  for (var n9 = new Array(t9), r10 = 0; r10 < t9; ++r10)
    n9[r10] = e10[r10];
  return n9;
}
var e, t, n, r, i, s, y;
var init_chunk_4bd36a8f = __esm({
  "node_modules/@jspm/core/nodelibs/browser/chunk-4bd36a8f.js"() {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    n = "object" == typeof Reflect ? Reflect : null;
    r = n && "function" == typeof n.apply ? n.apply : function(e10, t9, n9) {
      return Function.prototype.apply.call(e10, t9, n9);
    };
    t = n && "function" == typeof n.ownKeys ? n.ownKeys : Object.getOwnPropertySymbols ? function(e10) {
      return Object.getOwnPropertyNames(e10).concat(Object.getOwnPropertySymbols(e10));
    } : function(e10) {
      return Object.getOwnPropertyNames(e10);
    };
    i = Number.isNaN || function(e10) {
      return e10 != e10;
    };
    e = o, o.EventEmitter = o, o.prototype._events = void 0, o.prototype._eventsCount = 0, o.prototype._maxListeners = void 0;
    s = 10;
    Object.defineProperty(o, "defaultMaxListeners", { enumerable: true, get: function() {
      return s;
    }, set: function(e10) {
      if ("number" != typeof e10 || e10 < 0 || i(e10))
        throw new RangeError('The value of "defaultMaxListeners" is out of range. It must be a non-negative number. Received ' + e10 + ".");
      s = e10;
    } }), o.init = function() {
      void 0 !== this._events && this._events !== Object.getPrototypeOf(this)._events || (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0), this._maxListeners = this._maxListeners || void 0;
    }, o.prototype.setMaxListeners = function(e10) {
      if ("number" != typeof e10 || e10 < 0 || i(e10))
        throw new RangeError('The value of "n" is out of range. It must be a non-negative number. Received ' + e10 + ".");
      return this._maxListeners = e10, this;
    }, o.prototype.getMaxListeners = function() {
      return f(this);
    }, o.prototype.emit = function(e10) {
      for (var t9 = [], n9 = 1; n9 < arguments.length; n9++)
        t9.push(arguments[n9]);
      var i7 = "error" === e10, o9 = this._events;
      if (void 0 !== o9)
        i7 = i7 && void 0 === o9.error;
      else if (!i7)
        return false;
      if (i7) {
        var s6;
        if (t9.length > 0 && (s6 = t9[0]), s6 instanceof Error)
          throw s6;
        var u7 = new Error("Unhandled error." + (s6 ? " (" + s6.message + ")" : ""));
        throw u7.context = s6, u7;
      }
      var f7 = o9[e10];
      if (void 0 === f7)
        return false;
      if ("function" == typeof f7)
        r(f7, this, t9);
      else {
        var v7 = f7.length, a7 = c(f7, v7);
        for (n9 = 0; n9 < v7; ++n9)
          r(a7[n9], this, t9);
      }
      return true;
    }, o.prototype.addListener = function(e10, t9) {
      return v(this, e10, t9, false);
    }, o.prototype.on = o.prototype.addListener, o.prototype.prependListener = function(e10, t9) {
      return v(this, e10, t9, true);
    }, o.prototype.once = function(e10, t9) {
      return u(t9), this.on(e10, l(this, e10, t9)), this;
    }, o.prototype.prependOnceListener = function(e10, t9) {
      return u(t9), this.prependListener(e10, l(this, e10, t9)), this;
    }, o.prototype.removeListener = function(e10, t9) {
      var n9, r10, i7, o9, s6;
      if (u(t9), void 0 === (r10 = this._events))
        return this;
      if (void 0 === (n9 = r10[e10]))
        return this;
      if (n9 === t9 || n9.listener === t9)
        0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : (delete r10[e10], r10.removeListener && this.emit("removeListener", e10, n9.listener || t9));
      else if ("function" != typeof n9) {
        for (i7 = -1, o9 = n9.length - 1; o9 >= 0; o9--)
          if (n9[o9] === t9 || n9[o9].listener === t9) {
            s6 = n9[o9].listener, i7 = o9;
            break;
          }
        if (i7 < 0)
          return this;
        0 === i7 ? n9.shift() : !function(e11, t10) {
          for (; t10 + 1 < e11.length; t10++)
            e11[t10] = e11[t10 + 1];
          e11.pop();
        }(n9, i7), 1 === n9.length && (r10[e10] = n9[0]), void 0 !== r10.removeListener && this.emit("removeListener", e10, s6 || t9);
      }
      return this;
    }, o.prototype.off = o.prototype.removeListener, o.prototype.removeAllListeners = function(e10) {
      var t9, n9, r10;
      if (void 0 === (n9 = this._events))
        return this;
      if (void 0 === n9.removeListener)
        return 0 === arguments.length ? (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0) : void 0 !== n9[e10] && (0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : delete n9[e10]), this;
      if (0 === arguments.length) {
        var i7, o9 = Object.keys(n9);
        for (r10 = 0; r10 < o9.length; ++r10)
          "removeListener" !== (i7 = o9[r10]) && this.removeAllListeners(i7);
        return this.removeAllListeners("removeListener"), this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0, this;
      }
      if ("function" == typeof (t9 = n9[e10]))
        this.removeListener(e10, t9);
      else if (void 0 !== t9)
        for (r10 = t9.length - 1; r10 >= 0; r10--)
          this.removeListener(e10, t9[r10]);
      return this;
    }, o.prototype.listeners = function(e10) {
      return h(this, e10, true);
    }, o.prototype.rawListeners = function(e10) {
      return h(this, e10, false);
    }, o.listenerCount = function(e10, t9) {
      return "function" == typeof e10.listenerCount ? e10.listenerCount(t9) : p.call(e10, t9);
    }, o.prototype.listenerCount = p, o.prototype.eventNames = function() {
      return this._eventsCount > 0 ? t(this._events) : [];
    };
    y = e;
    y.EventEmitter;
    y.defaultMaxListeners;
    y.init;
    y.listenerCount;
    y.EventEmitter;
    y.defaultMaxListeners;
    y.init;
    y.listenerCount;
  }
});

// node_modules/@jspm/core/nodelibs/browser/events.js
var EventEmitter, defaultMaxListeners, init, listenerCount, on2, once2;
var init_events = __esm({
  "node_modules/@jspm/core/nodelibs/browser/events.js"() {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    init_chunk_4bd36a8f();
    init_chunk_4bd36a8f();
    y.once = function(emitter, event) {
      return new Promise((resolve4, reject) => {
        function eventListener(...args) {
          if (errorListener !== void 0) {
            emitter.removeListener("error", errorListener);
          }
          resolve4(args);
        }
        let errorListener;
        if (event !== "error") {
          errorListener = (err) => {
            emitter.removeListener(name, eventListener);
            reject(err);
          };
          emitter.once("error", errorListener);
        }
        emitter.once(event, eventListener);
      });
    };
    y.on = function(emitter, event) {
      const unconsumedEventValues = [];
      const unconsumedPromises = [];
      let error = null;
      let finished2 = false;
      const iterator = {
        async next() {
          const value = unconsumedEventValues.shift();
          if (value) {
            return createIterResult(value, false);
          }
          if (error) {
            const p7 = Promise.reject(error);
            error = null;
            return p7;
          }
          if (finished2) {
            return createIterResult(void 0, true);
          }
          return new Promise((resolve4, reject) => unconsumedPromises.push({ resolve: resolve4, reject }));
        },
        async return() {
          emitter.removeListener(event, eventHandler);
          emitter.removeListener("error", errorHandler);
          finished2 = true;
          for (const promise of unconsumedPromises) {
            promise.resolve(createIterResult(void 0, true));
          }
          return createIterResult(void 0, true);
        },
        throw(err) {
          error = err;
          emitter.removeListener(event, eventHandler);
          emitter.removeListener("error", errorHandler);
        },
        [Symbol.asyncIterator]() {
          return this;
        }
      };
      emitter.on(event, eventHandler);
      emitter.on("error", errorHandler);
      return iterator;
      function eventHandler(...args) {
        const promise = unconsumedPromises.shift();
        if (promise) {
          promise.resolve(createIterResult(args, false));
        } else {
          unconsumedEventValues.push(args);
        }
      }
      function errorHandler(err) {
        finished2 = true;
        const toError = unconsumedPromises.shift();
        if (toError) {
          toError.reject(err);
        } else {
          error = err;
        }
        iterator.return();
      }
    };
    ({
      EventEmitter,
      defaultMaxListeners,
      init,
      listenerCount,
      on: on2,
      once: once2
    } = y);
  }
});

// node_modules/object-keys/isArguments.js
var require_isArguments = __commonJS({
  "node_modules/object-keys/isArguments.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var toStr = Object.prototype.toString;
    module.exports = function isArguments(value) {
      var str = toStr.call(value);
      var isArgs = str === "[object Arguments]";
      if (!isArgs) {
        isArgs = str !== "[object Array]" && value !== null && typeof value === "object" && typeof value.length === "number" && value.length >= 0 && toStr.call(value.callee) === "[object Function]";
      }
      return isArgs;
    };
  }
});

// node_modules/object-keys/implementation.js
var require_implementation = __commonJS({
  "node_modules/object-keys/implementation.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var keysShim;
    if (!Object.keys) {
      has = Object.prototype.hasOwnProperty;
      toStr = Object.prototype.toString;
      isArgs = require_isArguments();
      isEnumerable = Object.prototype.propertyIsEnumerable;
      hasDontEnumBug = !isEnumerable.call({ toString: null }, "toString");
      hasProtoEnumBug = isEnumerable.call(function() {
      }, "prototype");
      dontEnums = [
        "toString",
        "toLocaleString",
        "valueOf",
        "hasOwnProperty",
        "isPrototypeOf",
        "propertyIsEnumerable",
        "constructor"
      ];
      equalsConstructorPrototype = function(o9) {
        var ctor = o9.constructor;
        return ctor && ctor.prototype === o9;
      };
      excludedKeys = {
        $applicationCache: true,
        $console: true,
        $external: true,
        $frame: true,
        $frameElement: true,
        $frames: true,
        $innerHeight: true,
        $innerWidth: true,
        $onmozfullscreenchange: true,
        $onmozfullscreenerror: true,
        $outerHeight: true,
        $outerWidth: true,
        $pageXOffset: true,
        $pageYOffset: true,
        $parent: true,
        $scrollLeft: true,
        $scrollTop: true,
        $scrollX: true,
        $scrollY: true,
        $self: true,
        $webkitIndexedDB: true,
        $webkitStorageInfo: true,
        $window: true
      };
      hasAutomationEqualityBug = function() {
        if (typeof window === "undefined") {
          return false;
        }
        for (var k4 in window) {
          try {
            if (!excludedKeys["$" + k4] && has.call(window, k4) && window[k4] !== null && typeof window[k4] === "object") {
              try {
                equalsConstructorPrototype(window[k4]);
              } catch (e10) {
                return true;
              }
            }
          } catch (e10) {
            return true;
          }
        }
        return false;
      }();
      equalsConstructorPrototypeIfNotBuggy = function(o9) {
        if (typeof window === "undefined" || !hasAutomationEqualityBug) {
          return equalsConstructorPrototype(o9);
        }
        try {
          return equalsConstructorPrototype(o9);
        } catch (e10) {
          return false;
        }
      };
      keysShim = function keys(object) {
        var isObject4 = object !== null && typeof object === "object";
        var isFunction4 = toStr.call(object) === "[object Function]";
        var isArguments = isArgs(object);
        var isString5 = isObject4 && toStr.call(object) === "[object String]";
        var theKeys = [];
        if (!isObject4 && !isFunction4 && !isArguments) {
          throw new TypeError("Object.keys called on a non-object");
        }
        var skipProto = hasProtoEnumBug && isFunction4;
        if (isString5 && object.length > 0 && !has.call(object, 0)) {
          for (var i7 = 0; i7 < object.length; ++i7) {
            theKeys.push(String(i7));
          }
        }
        if (isArguments && object.length > 0) {
          for (var j4 = 0; j4 < object.length; ++j4) {
            theKeys.push(String(j4));
          }
        } else {
          for (var name3 in object) {
            if (!(skipProto && name3 === "prototype") && has.call(object, name3)) {
              theKeys.push(String(name3));
            }
          }
        }
        if (hasDontEnumBug) {
          var skipConstructor = equalsConstructorPrototypeIfNotBuggy(object);
          for (var k4 = 0; k4 < dontEnums.length; ++k4) {
            if (!(skipConstructor && dontEnums[k4] === "constructor") && has.call(object, dontEnums[k4])) {
              theKeys.push(dontEnums[k4]);
            }
          }
        }
        return theKeys;
      };
    }
    var has;
    var toStr;
    var isArgs;
    var isEnumerable;
    var hasDontEnumBug;
    var hasProtoEnumBug;
    var dontEnums;
    var equalsConstructorPrototype;
    var excludedKeys;
    var hasAutomationEqualityBug;
    var equalsConstructorPrototypeIfNotBuggy;
    module.exports = keysShim;
  }
});

// node_modules/object-keys/index.js
var require_object_keys = __commonJS({
  "node_modules/object-keys/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var slice = Array.prototype.slice;
    var isArgs = require_isArguments();
    var origKeys = Object.keys;
    var keysShim = origKeys ? function keys(o9) {
      return origKeys(o9);
    } : require_implementation();
    var originalKeys = Object.keys;
    keysShim.shim = function shimObjectKeys() {
      if (Object.keys) {
        var keysWorksWithArguments = function() {
          var args = Object.keys(arguments);
          return args && args.length === arguments.length;
        }(1, 2);
        if (!keysWorksWithArguments) {
          Object.keys = function keys(object) {
            if (isArgs(object)) {
              return originalKeys(slice.call(object));
            }
            return originalKeys(object);
          };
        }
      } else {
        Object.keys = keysShim;
      }
      return Object.keys || keysShim;
    };
    module.exports = keysShim;
  }
});

// node_modules/has-symbols/shams.js
var require_shams = __commonJS({
  "node_modules/has-symbols/shams.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    module.exports = function hasSymbols() {
      if (typeof Symbol !== "function" || typeof Object.getOwnPropertySymbols !== "function") {
        return false;
      }
      if (typeof Symbol.iterator === "symbol") {
        return true;
      }
      var obj = {};
      var sym = Symbol("test");
      var symObj = Object(sym);
      if (typeof sym === "string") {
        return false;
      }
      if (Object.prototype.toString.call(sym) !== "[object Symbol]") {
        return false;
      }
      if (Object.prototype.toString.call(symObj) !== "[object Symbol]") {
        return false;
      }
      var symVal = 42;
      obj[sym] = symVal;
      for (sym in obj) {
        return false;
      }
      if (typeof Object.keys === "function" && Object.keys(obj).length !== 0) {
        return false;
      }
      if (typeof Object.getOwnPropertyNames === "function" && Object.getOwnPropertyNames(obj).length !== 0) {
        return false;
      }
      var syms = Object.getOwnPropertySymbols(obj);
      if (syms.length !== 1 || syms[0] !== sym) {
        return false;
      }
      if (!Object.prototype.propertyIsEnumerable.call(obj, sym)) {
        return false;
      }
      if (typeof Object.getOwnPropertyDescriptor === "function") {
        var descriptor = Object.getOwnPropertyDescriptor(obj, sym);
        if (descriptor.value !== symVal || descriptor.enumerable !== true) {
          return false;
        }
      }
      return true;
    };
  }
});

// node_modules/has-symbols/index.js
var require_has_symbols = __commonJS({
  "node_modules/has-symbols/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var origSymbol = typeof Symbol !== "undefined" && Symbol;
    var hasSymbolSham = require_shams();
    module.exports = function hasNativeSymbols() {
      if (typeof origSymbol !== "function") {
        return false;
      }
      if (typeof Symbol !== "function") {
        return false;
      }
      if (typeof origSymbol("foo") !== "symbol") {
        return false;
      }
      if (typeof Symbol("bar") !== "symbol") {
        return false;
      }
      return hasSymbolSham();
    };
  }
});

// node_modules/has-proto/index.js
var require_has_proto = __commonJS({
  "node_modules/has-proto/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var test = {
      foo: {}
    };
    var $Object = Object;
    module.exports = function hasProto() {
      return { __proto__: test }.foo === test.foo && !({ __proto__: null } instanceof $Object);
    };
  }
});

// node_modules/function-bind/implementation.js
var require_implementation2 = __commonJS({
  "node_modules/function-bind/implementation.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var ERROR_MESSAGE = "Function.prototype.bind called on incompatible ";
    var slice = Array.prototype.slice;
    var toStr = Object.prototype.toString;
    var funcType = "[object Function]";
    module.exports = function bind2(that) {
      var target = this;
      if (typeof target !== "function" || toStr.call(target) !== funcType) {
        throw new TypeError(ERROR_MESSAGE + target);
      }
      var args = slice.call(arguments, 1);
      var bound;
      var binder = function() {
        if (this instanceof bound) {
          var result = target.apply(
            this,
            args.concat(slice.call(arguments))
          );
          if (Object(result) === result) {
            return result;
          }
          return this;
        } else {
          return target.apply(
            that,
            args.concat(slice.call(arguments))
          );
        }
      };
      var boundLength = Math.max(0, target.length - args.length);
      var boundArgs = [];
      for (var i7 = 0; i7 < boundLength; i7++) {
        boundArgs.push("$" + i7);
      }
      bound = Function("binder", "return function (" + boundArgs.join(",") + "){ return binder.apply(this,arguments); }")(binder);
      if (target.prototype) {
        var Empty = function Empty2() {
        };
        Empty.prototype = target.prototype;
        bound.prototype = new Empty();
        Empty.prototype = null;
      }
      return bound;
    };
  }
});

// node_modules/function-bind/index.js
var require_function_bind = __commonJS({
  "node_modules/function-bind/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var implementation = require_implementation2();
    module.exports = Function.prototype.bind || implementation;
  }
});

// node_modules/has/src/index.js
var require_src = __commonJS({
  "node_modules/has/src/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var bind2 = require_function_bind();
    module.exports = bind2.call(Function.call, Object.prototype.hasOwnProperty);
  }
});

// node_modules/get-intrinsic/index.js
var require_get_intrinsic = __commonJS({
  "node_modules/get-intrinsic/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var undefined2;
    var $SyntaxError = SyntaxError;
    var $Function = Function;
    var $TypeError = TypeError;
    var getEvalledConstructor = function(expressionSyntax) {
      try {
        return $Function('"use strict"; return (' + expressionSyntax + ").constructor;")();
      } catch (e10) {
      }
    };
    var $gOPD = Object.getOwnPropertyDescriptor;
    if ($gOPD) {
      try {
        $gOPD({}, "");
      } catch (e10) {
        $gOPD = null;
      }
    }
    var throwTypeError = function() {
      throw new $TypeError();
    };
    var ThrowTypeError = $gOPD ? function() {
      try {
        arguments.callee;
        return throwTypeError;
      } catch (calleeThrows) {
        try {
          return $gOPD(arguments, "callee").get;
        } catch (gOPDthrows) {
          return throwTypeError;
        }
      }
    }() : throwTypeError;
    var hasSymbols = require_has_symbols()();
    var hasProto = require_has_proto()();
    var getProto = Object.getPrototypeOf || (hasProto ? function(x4) {
      return x4.__proto__;
    } : null);
    var needsEval = {};
    var TypedArray = typeof Uint8Array === "undefined" || !getProto ? undefined2 : getProto(Uint8Array);
    var INTRINSICS = {
      "%AggregateError%": typeof AggregateError === "undefined" ? undefined2 : AggregateError,
      "%Array%": Array,
      "%ArrayBuffer%": typeof ArrayBuffer === "undefined" ? undefined2 : ArrayBuffer,
      "%ArrayIteratorPrototype%": hasSymbols && getProto ? getProto([][Symbol.iterator]()) : undefined2,
      "%AsyncFromSyncIteratorPrototype%": undefined2,
      "%AsyncFunction%": needsEval,
      "%AsyncGenerator%": needsEval,
      "%AsyncGeneratorFunction%": needsEval,
      "%AsyncIteratorPrototype%": needsEval,
      "%Atomics%": typeof Atomics === "undefined" ? undefined2 : Atomics,
      "%BigInt%": typeof BigInt === "undefined" ? undefined2 : BigInt,
      "%BigInt64Array%": typeof BigInt64Array === "undefined" ? undefined2 : BigInt64Array,
      "%BigUint64Array%": typeof BigUint64Array === "undefined" ? undefined2 : BigUint64Array,
      "%Boolean%": Boolean,
      "%DataView%": typeof DataView === "undefined" ? undefined2 : DataView,
      "%Date%": Date,
      "%decodeURI%": decodeURI,
      "%decodeURIComponent%": decodeURIComponent,
      "%encodeURI%": encodeURI,
      "%encodeURIComponent%": encodeURIComponent,
      "%Error%": Error,
      "%eval%": eval,
      // eslint-disable-line no-eval
      "%EvalError%": EvalError,
      "%Float32Array%": typeof Float32Array === "undefined" ? undefined2 : Float32Array,
      "%Float64Array%": typeof Float64Array === "undefined" ? undefined2 : Float64Array,
      "%FinalizationRegistry%": typeof FinalizationRegistry === "undefined" ? undefined2 : FinalizationRegistry,
      "%Function%": $Function,
      "%GeneratorFunction%": needsEval,
      "%Int8Array%": typeof Int8Array === "undefined" ? undefined2 : Int8Array,
      "%Int16Array%": typeof Int16Array === "undefined" ? undefined2 : Int16Array,
      "%Int32Array%": typeof Int32Array === "undefined" ? undefined2 : Int32Array,
      "%isFinite%": isFinite,
      "%isNaN%": isNaN,
      "%IteratorPrototype%": hasSymbols && getProto ? getProto(getProto([][Symbol.iterator]())) : undefined2,
      "%JSON%": typeof JSON === "object" ? JSON : undefined2,
      "%Map%": typeof Map === "undefined" ? undefined2 : Map,
      "%MapIteratorPrototype%": typeof Map === "undefined" || !hasSymbols || !getProto ? undefined2 : getProto((/* @__PURE__ */ new Map())[Symbol.iterator]()),
      "%Math%": Math,
      "%Number%": Number,
      "%Object%": Object,
      "%parseFloat%": parseFloat,
      "%parseInt%": parseInt,
      "%Promise%": typeof Promise === "undefined" ? undefined2 : Promise,
      "%Proxy%": typeof Proxy === "undefined" ? undefined2 : Proxy,
      "%RangeError%": RangeError,
      "%ReferenceError%": ReferenceError,
      "%Reflect%": typeof Reflect === "undefined" ? undefined2 : Reflect,
      "%RegExp%": RegExp,
      "%Set%": typeof Set === "undefined" ? undefined2 : Set,
      "%SetIteratorPrototype%": typeof Set === "undefined" || !hasSymbols || !getProto ? undefined2 : getProto((/* @__PURE__ */ new Set())[Symbol.iterator]()),
      "%SharedArrayBuffer%": typeof SharedArrayBuffer === "undefined" ? undefined2 : SharedArrayBuffer,
      "%String%": String,
      "%StringIteratorPrototype%": hasSymbols && getProto ? getProto(""[Symbol.iterator]()) : undefined2,
      "%Symbol%": hasSymbols ? Symbol : undefined2,
      "%SyntaxError%": $SyntaxError,
      "%ThrowTypeError%": ThrowTypeError,
      "%TypedArray%": TypedArray,
      "%TypeError%": $TypeError,
      "%Uint8Array%": typeof Uint8Array === "undefined" ? undefined2 : Uint8Array,
      "%Uint8ClampedArray%": typeof Uint8ClampedArray === "undefined" ? undefined2 : Uint8ClampedArray,
      "%Uint16Array%": typeof Uint16Array === "undefined" ? undefined2 : Uint16Array,
      "%Uint32Array%": typeof Uint32Array === "undefined" ? undefined2 : Uint32Array,
      "%URIError%": URIError,
      "%WeakMap%": typeof WeakMap === "undefined" ? undefined2 : WeakMap,
      "%WeakRef%": typeof WeakRef === "undefined" ? undefined2 : WeakRef,
      "%WeakSet%": typeof WeakSet === "undefined" ? undefined2 : WeakSet
    };
    if (getProto) {
      try {
        null.error;
      } catch (e10) {
        errorProto = getProto(getProto(e10));
        INTRINSICS["%Error.prototype%"] = errorProto;
      }
    }
    var errorProto;
    var doEval = function doEval2(name3) {
      var value;
      if (name3 === "%AsyncFunction%") {
        value = getEvalledConstructor("async function () {}");
      } else if (name3 === "%GeneratorFunction%") {
        value = getEvalledConstructor("function* () {}");
      } else if (name3 === "%AsyncGeneratorFunction%") {
        value = getEvalledConstructor("async function* () {}");
      } else if (name3 === "%AsyncGenerator%") {
        var fn = doEval2("%AsyncGeneratorFunction%");
        if (fn) {
          value = fn.prototype;
        }
      } else if (name3 === "%AsyncIteratorPrototype%") {
        var gen = doEval2("%AsyncGenerator%");
        if (gen && getProto) {
          value = getProto(gen.prototype);
        }
      }
      INTRINSICS[name3] = value;
      return value;
    };
    var LEGACY_ALIASES = {
      "%ArrayBufferPrototype%": ["ArrayBuffer", "prototype"],
      "%ArrayPrototype%": ["Array", "prototype"],
      "%ArrayProto_entries%": ["Array", "prototype", "entries"],
      "%ArrayProto_forEach%": ["Array", "prototype", "forEach"],
      "%ArrayProto_keys%": ["Array", "prototype", "keys"],
      "%ArrayProto_values%": ["Array", "prototype", "values"],
      "%AsyncFunctionPrototype%": ["AsyncFunction", "prototype"],
      "%AsyncGenerator%": ["AsyncGeneratorFunction", "prototype"],
      "%AsyncGeneratorPrototype%": ["AsyncGeneratorFunction", "prototype", "prototype"],
      "%BooleanPrototype%": ["Boolean", "prototype"],
      "%DataViewPrototype%": ["DataView", "prototype"],
      "%DatePrototype%": ["Date", "prototype"],
      "%ErrorPrototype%": ["Error", "prototype"],
      "%EvalErrorPrototype%": ["EvalError", "prototype"],
      "%Float32ArrayPrototype%": ["Float32Array", "prototype"],
      "%Float64ArrayPrototype%": ["Float64Array", "prototype"],
      "%FunctionPrototype%": ["Function", "prototype"],
      "%Generator%": ["GeneratorFunction", "prototype"],
      "%GeneratorPrototype%": ["GeneratorFunction", "prototype", "prototype"],
      "%Int8ArrayPrototype%": ["Int8Array", "prototype"],
      "%Int16ArrayPrototype%": ["Int16Array", "prototype"],
      "%Int32ArrayPrototype%": ["Int32Array", "prototype"],
      "%JSONParse%": ["JSON", "parse"],
      "%JSONStringify%": ["JSON", "stringify"],
      "%MapPrototype%": ["Map", "prototype"],
      "%NumberPrototype%": ["Number", "prototype"],
      "%ObjectPrototype%": ["Object", "prototype"],
      "%ObjProto_toString%": ["Object", "prototype", "toString"],
      "%ObjProto_valueOf%": ["Object", "prototype", "valueOf"],
      "%PromisePrototype%": ["Promise", "prototype"],
      "%PromiseProto_then%": ["Promise", "prototype", "then"],
      "%Promise_all%": ["Promise", "all"],
      "%Promise_reject%": ["Promise", "reject"],
      "%Promise_resolve%": ["Promise", "resolve"],
      "%RangeErrorPrototype%": ["RangeError", "prototype"],
      "%ReferenceErrorPrototype%": ["ReferenceError", "prototype"],
      "%RegExpPrototype%": ["RegExp", "prototype"],
      "%SetPrototype%": ["Set", "prototype"],
      "%SharedArrayBufferPrototype%": ["SharedArrayBuffer", "prototype"],
      "%StringPrototype%": ["String", "prototype"],
      "%SymbolPrototype%": ["Symbol", "prototype"],
      "%SyntaxErrorPrototype%": ["SyntaxError", "prototype"],
      "%TypedArrayPrototype%": ["TypedArray", "prototype"],
      "%TypeErrorPrototype%": ["TypeError", "prototype"],
      "%Uint8ArrayPrototype%": ["Uint8Array", "prototype"],
      "%Uint8ClampedArrayPrototype%": ["Uint8ClampedArray", "prototype"],
      "%Uint16ArrayPrototype%": ["Uint16Array", "prototype"],
      "%Uint32ArrayPrototype%": ["Uint32Array", "prototype"],
      "%URIErrorPrototype%": ["URIError", "prototype"],
      "%WeakMapPrototype%": ["WeakMap", "prototype"],
      "%WeakSetPrototype%": ["WeakSet", "prototype"]
    };
    var bind2 = require_function_bind();
    var hasOwn = require_src();
    var $concat = bind2.call(Function.call, Array.prototype.concat);
    var $spliceApply = bind2.call(Function.apply, Array.prototype.splice);
    var $replace = bind2.call(Function.call, String.prototype.replace);
    var $strSlice = bind2.call(Function.call, String.prototype.slice);
    var $exec = bind2.call(Function.call, RegExp.prototype.exec);
    var rePropName = /[^%.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|%$))/g;
    var reEscapeChar = /\\(\\)?/g;
    var stringToPath = function stringToPath2(string) {
      var first = $strSlice(string, 0, 1);
      var last = $strSlice(string, -1);
      if (first === "%" && last !== "%") {
        throw new $SyntaxError("invalid intrinsic syntax, expected closing `%`");
      } else if (last === "%" && first !== "%") {
        throw new $SyntaxError("invalid intrinsic syntax, expected opening `%`");
      }
      var result = [];
      $replace(string, rePropName, function(match, number, quote, subString) {
        result[result.length] = quote ? $replace(subString, reEscapeChar, "$1") : number || match;
      });
      return result;
    };
    var getBaseIntrinsic = function getBaseIntrinsic2(name3, allowMissing) {
      var intrinsicName = name3;
      var alias;
      if (hasOwn(LEGACY_ALIASES, intrinsicName)) {
        alias = LEGACY_ALIASES[intrinsicName];
        intrinsicName = "%" + alias[0] + "%";
      }
      if (hasOwn(INTRINSICS, intrinsicName)) {
        var value = INTRINSICS[intrinsicName];
        if (value === needsEval) {
          value = doEval(intrinsicName);
        }
        if (typeof value === "undefined" && !allowMissing) {
          throw new $TypeError("intrinsic " + name3 + " exists, but is not available. Please file an issue!");
        }
        return {
          alias,
          name: intrinsicName,
          value
        };
      }
      throw new $SyntaxError("intrinsic " + name3 + " does not exist!");
    };
    module.exports = function GetIntrinsic(name3, allowMissing) {
      if (typeof name3 !== "string" || name3.length === 0) {
        throw new $TypeError("intrinsic name must be a non-empty string");
      }
      if (arguments.length > 1 && typeof allowMissing !== "boolean") {
        throw new $TypeError('"allowMissing" argument must be a boolean');
      }
      if ($exec(/^%?[^%]*%?$/, name3) === null) {
        throw new $SyntaxError("`%` may not be present anywhere but at the beginning and end of the intrinsic name");
      }
      var parts = stringToPath(name3);
      var intrinsicBaseName = parts.length > 0 ? parts[0] : "";
      var intrinsic = getBaseIntrinsic("%" + intrinsicBaseName + "%", allowMissing);
      var intrinsicRealName = intrinsic.name;
      var value = intrinsic.value;
      var skipFurtherCaching = false;
      var alias = intrinsic.alias;
      if (alias) {
        intrinsicBaseName = alias[0];
        $spliceApply(parts, $concat([0, 1], alias));
      }
      for (var i7 = 1, isOwn = true; i7 < parts.length; i7 += 1) {
        var part = parts[i7];
        var first = $strSlice(part, 0, 1);
        var last = $strSlice(part, -1);
        if ((first === '"' || first === "'" || first === "`" || (last === '"' || last === "'" || last === "`")) && first !== last) {
          throw new $SyntaxError("property names with quotes must have matching quotes");
        }
        if (part === "constructor" || !isOwn) {
          skipFurtherCaching = true;
        }
        intrinsicBaseName += "." + part;
        intrinsicRealName = "%" + intrinsicBaseName + "%";
        if (hasOwn(INTRINSICS, intrinsicRealName)) {
          value = INTRINSICS[intrinsicRealName];
        } else if (value != null) {
          if (!(part in value)) {
            if (!allowMissing) {
              throw new $TypeError("base intrinsic for " + name3 + " exists, but the property is not available.");
            }
            return void 0;
          }
          if ($gOPD && i7 + 1 >= parts.length) {
            var desc = $gOPD(value, part);
            isOwn = !!desc;
            if (isOwn && "get" in desc && !("originalValue" in desc.get)) {
              value = desc.get;
            } else {
              value = value[part];
            }
          } else {
            isOwn = hasOwn(value, part);
            value = value[part];
          }
          if (isOwn && !skipFurtherCaching) {
            INTRINSICS[intrinsicRealName] = value;
          }
        }
      }
      return value;
    };
  }
});

// node_modules/has-property-descriptors/index.js
var require_has_property_descriptors = __commonJS({
  "node_modules/has-property-descriptors/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var $defineProperty = GetIntrinsic("%Object.defineProperty%", true);
    var hasPropertyDescriptors = function hasPropertyDescriptors2() {
      if ($defineProperty) {
        try {
          $defineProperty({}, "a", { value: 1 });
          return true;
        } catch (e10) {
          return false;
        }
      }
      return false;
    };
    hasPropertyDescriptors.hasArrayLengthDefineBug = function hasArrayLengthDefineBug() {
      if (!hasPropertyDescriptors()) {
        return null;
      }
      try {
        return $defineProperty([], "length", { value: 1 }).length !== 1;
      } catch (e10) {
        return true;
      }
    };
    module.exports = hasPropertyDescriptors;
  }
});

// node_modules/define-properties/index.js
var require_define_properties = __commonJS({
  "node_modules/define-properties/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var keys = require_object_keys();
    var hasSymbols = typeof Symbol === "function" && typeof Symbol("foo") === "symbol";
    var toStr = Object.prototype.toString;
    var concat = Array.prototype.concat;
    var origDefineProperty = Object.defineProperty;
    var isFunction4 = function(fn) {
      return typeof fn === "function" && toStr.call(fn) === "[object Function]";
    };
    var hasPropertyDescriptors = require_has_property_descriptors()();
    var supportsDescriptors = origDefineProperty && hasPropertyDescriptors;
    var defineProperty = function(object, name3, value, predicate) {
      if (name3 in object) {
        if (predicate === true) {
          if (object[name3] === value) {
            return;
          }
        } else if (!isFunction4(predicate) || !predicate()) {
          return;
        }
      }
      if (supportsDescriptors) {
        origDefineProperty(object, name3, {
          configurable: true,
          enumerable: false,
          value,
          writable: true
        });
      } else {
        object[name3] = value;
      }
    };
    var defineProperties = function(object, map) {
      var predicates2 = arguments.length > 2 ? arguments[2] : {};
      var props = keys(map);
      if (hasSymbols) {
        props = concat.call(props, Object.getOwnPropertySymbols(map));
      }
      for (var i7 = 0; i7 < props.length; i7 += 1) {
        defineProperty(object, props[i7], map[props[i7]], predicates2[props[i7]]);
      }
    };
    defineProperties.supportsDescriptors = !!supportsDescriptors;
    module.exports = defineProperties;
  }
});

// node_modules/call-bind/index.js
var require_call_bind = __commonJS({
  "node_modules/call-bind/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var bind2 = require_function_bind();
    var GetIntrinsic = require_get_intrinsic();
    var $apply = GetIntrinsic("%Function.prototype.apply%");
    var $call = GetIntrinsic("%Function.prototype.call%");
    var $reflectApply = GetIntrinsic("%Reflect.apply%", true) || bind2.call($call, $apply);
    var $gOPD = GetIntrinsic("%Object.getOwnPropertyDescriptor%", true);
    var $defineProperty = GetIntrinsic("%Object.defineProperty%", true);
    var $max = GetIntrinsic("%Math.max%");
    if ($defineProperty) {
      try {
        $defineProperty({}, "a", { value: 1 });
      } catch (e10) {
        $defineProperty = null;
      }
    }
    module.exports = function callBind(originalFunction) {
      var func = $reflectApply(bind2, $call, arguments);
      if ($gOPD && $defineProperty) {
        var desc = $gOPD(func, "length");
        if (desc.configurable) {
          $defineProperty(
            func,
            "length",
            { value: 1 + $max(0, originalFunction.length - (arguments.length - 1)) }
          );
        }
      }
      return func;
    };
    var applyBind = function applyBind2() {
      return $reflectApply(bind2, $apply, arguments);
    };
    if ($defineProperty) {
      $defineProperty(module.exports, "apply", { value: applyBind });
    } else {
      module.exports.apply = applyBind;
    }
  }
});

// node_modules/call-bind/callBound.js
var require_callBound = __commonJS({
  "node_modules/call-bind/callBound.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var callBind = require_call_bind();
    var $indexOf = callBind(GetIntrinsic("String.prototype.indexOf"));
    module.exports = function callBoundIntrinsic(name3, allowMissing) {
      var intrinsic = GetIntrinsic(name3, !!allowMissing);
      if (typeof intrinsic === "function" && $indexOf(name3, ".prototype.") > -1) {
        return callBind(intrinsic);
      }
      return intrinsic;
    };
  }
});

// node_modules/object.assign/implementation.js
var require_implementation3 = __commonJS({
  "node_modules/object.assign/implementation.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var objectKeys = require_object_keys();
    var hasSymbols = require_shams()();
    var callBound = require_callBound();
    var toObject = Object;
    var $push = callBound("Array.prototype.push");
    var $propIsEnumerable = callBound("Object.prototype.propertyIsEnumerable");
    var originalGetSymbols = hasSymbols ? Object.getOwnPropertySymbols : null;
    module.exports = function assign(target, source1) {
      if (target == null) {
        throw new TypeError("target must be an object");
      }
      var to = toObject(target);
      if (arguments.length === 1) {
        return to;
      }
      for (var s6 = 1; s6 < arguments.length; ++s6) {
        var from = toObject(arguments[s6]);
        var keys = objectKeys(from);
        var getSymbols = hasSymbols && (Object.getOwnPropertySymbols || originalGetSymbols);
        if (getSymbols) {
          var syms = getSymbols(from);
          for (var j4 = 0; j4 < syms.length; ++j4) {
            var key = syms[j4];
            if ($propIsEnumerable(from, key)) {
              $push(keys, key);
            }
          }
        }
        for (var i7 = 0; i7 < keys.length; ++i7) {
          var nextKey = keys[i7];
          if ($propIsEnumerable(from, nextKey)) {
            var propValue = from[nextKey];
            to[nextKey] = propValue;
          }
        }
      }
      return to;
    };
  }
});

// node_modules/object.assign/polyfill.js
var require_polyfill = __commonJS({
  "node_modules/object.assign/polyfill.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var implementation = require_implementation3();
    var lacksProperEnumerationOrder = function() {
      if (!Object.assign) {
        return false;
      }
      var str = "abcdefghijklmnopqrst";
      var letters = str.split("");
      var map = {};
      for (var i7 = 0; i7 < letters.length; ++i7) {
        map[letters[i7]] = letters[i7];
      }
      var obj = Object.assign({}, map);
      var actual = "";
      for (var k4 in obj) {
        actual += k4;
      }
      return str !== actual;
    };
    var assignHasPendingExceptions = function() {
      if (!Object.assign || !Object.preventExtensions) {
        return false;
      }
      var thrower = Object.preventExtensions({ 1: 2 });
      try {
        Object.assign(thrower, "xy");
      } catch (e10) {
        return thrower[1] === "y";
      }
      return false;
    };
    module.exports = function getPolyfill() {
      if (!Object.assign) {
        return implementation;
      }
      if (lacksProperEnumerationOrder()) {
        return implementation;
      }
      if (assignHasPendingExceptions()) {
        return implementation;
      }
      return Object.assign;
    };
  }
});

// node_modules/object.assign/shim.js
var require_shim = __commonJS({
  "node_modules/object.assign/shim.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var define2 = require_define_properties();
    var getPolyfill = require_polyfill();
    module.exports = function shimAssign() {
      var polyfill = getPolyfill();
      define2(
        Object,
        { assign: polyfill },
        { assign: function() {
          return Object.assign !== polyfill;
        } }
      );
      return polyfill;
    };
  }
});

// node_modules/object.assign/index.js
var require_object = __commonJS({
  "node_modules/object.assign/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var defineProperties = require_define_properties();
    var callBind = require_call_bind();
    var implementation = require_implementation3();
    var getPolyfill = require_polyfill();
    var shim = require_shim();
    var polyfill = callBind.apply(getPolyfill());
    var bound = function assign(target, source1) {
      return polyfill(Object, arguments);
    };
    defineProperties(bound, {
      getPolyfill,
      implementation,
      shim
    });
    module.exports = bound;
  }
});

// node_modules/functions-have-names/index.js
var require_functions_have_names = __commonJS({
  "node_modules/functions-have-names/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var functionsHaveNames = function functionsHaveNames2() {
      return typeof function f7() {
      }.name === "string";
    };
    var gOPD = Object.getOwnPropertyDescriptor;
    if (gOPD) {
      try {
        gOPD([], "length");
      } catch (e10) {
        gOPD = null;
      }
    }
    functionsHaveNames.functionsHaveConfigurableNames = function functionsHaveConfigurableNames() {
      if (!functionsHaveNames() || !gOPD) {
        return false;
      }
      var desc = gOPD(function() {
      }, "name");
      return !!desc && !!desc.configurable;
    };
    var $bind = Function.prototype.bind;
    functionsHaveNames.boundFunctionsHaveNames = function boundFunctionsHaveNames() {
      return functionsHaveNames() && typeof $bind === "function" && function f7() {
      }.bind().name !== "";
    };
    module.exports = functionsHaveNames;
  }
});

// node_modules/regexp.prototype.flags/implementation.js
var require_implementation4 = __commonJS({
  "node_modules/regexp.prototype.flags/implementation.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var functionsHaveConfigurableNames = require_functions_have_names().functionsHaveConfigurableNames();
    var $Object = Object;
    var $TypeError = TypeError;
    module.exports = function flags() {
      if (this != null && this !== $Object(this)) {
        throw new $TypeError("RegExp.prototype.flags getter called on non-object");
      }
      var result = "";
      if (this.hasIndices) {
        result += "d";
      }
      if (this.global) {
        result += "g";
      }
      if (this.ignoreCase) {
        result += "i";
      }
      if (this.multiline) {
        result += "m";
      }
      if (this.dotAll) {
        result += "s";
      }
      if (this.unicode) {
        result += "u";
      }
      if (this.unicodeSets) {
        result += "v";
      }
      if (this.sticky) {
        result += "y";
      }
      return result;
    };
    if (functionsHaveConfigurableNames && Object.defineProperty) {
      Object.defineProperty(module.exports, "name", { value: "get flags" });
    }
  }
});

// node_modules/regexp.prototype.flags/polyfill.js
var require_polyfill2 = __commonJS({
  "node_modules/regexp.prototype.flags/polyfill.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var implementation = require_implementation4();
    var supportsDescriptors = require_define_properties().supportsDescriptors;
    var $gOPD = Object.getOwnPropertyDescriptor;
    module.exports = function getPolyfill() {
      if (supportsDescriptors && /a/mig.flags === "gim") {
        var descriptor = $gOPD(RegExp.prototype, "flags");
        if (descriptor && typeof descriptor.get === "function" && typeof RegExp.prototype.dotAll === "boolean" && typeof RegExp.prototype.hasIndices === "boolean") {
          var calls = "";
          var o9 = {};
          Object.defineProperty(o9, "hasIndices", {
            get: function() {
              calls += "d";
            }
          });
          Object.defineProperty(o9, "sticky", {
            get: function() {
              calls += "y";
            }
          });
          if (calls === "dy") {
            return descriptor.get;
          }
        }
      }
      return implementation;
    };
  }
});

// node_modules/regexp.prototype.flags/shim.js
var require_shim2 = __commonJS({
  "node_modules/regexp.prototype.flags/shim.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var supportsDescriptors = require_define_properties().supportsDescriptors;
    var getPolyfill = require_polyfill2();
    var gOPD = Object.getOwnPropertyDescriptor;
    var defineProperty = Object.defineProperty;
    var TypeErr = TypeError;
    var getProto = Object.getPrototypeOf;
    var regex = /a/;
    module.exports = function shimFlags() {
      if (!supportsDescriptors || !getProto) {
        throw new TypeErr("RegExp.prototype.flags requires a true ES5 environment that supports property descriptors");
      }
      var polyfill = getPolyfill();
      var proto = getProto(regex);
      var descriptor = gOPD(proto, "flags");
      if (!descriptor || descriptor.get !== polyfill) {
        defineProperty(proto, "flags", {
          configurable: true,
          enumerable: false,
          get: polyfill
        });
      }
      return polyfill;
    };
  }
});

// node_modules/regexp.prototype.flags/index.js
var require_regexp_prototype = __commonJS({
  "node_modules/regexp.prototype.flags/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var define2 = require_define_properties();
    var callBind = require_call_bind();
    var implementation = require_implementation4();
    var getPolyfill = require_polyfill2();
    var shim = require_shim2();
    var flagsBound = callBind(getPolyfill());
    define2(flagsBound, {
      getPolyfill,
      implementation,
      shim
    });
    module.exports = flagsBound;
  }
});

// node_modules/has-tostringtag/shams.js
var require_shams2 = __commonJS({
  "node_modules/has-tostringtag/shams.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var hasSymbols = require_shams();
    module.exports = function hasToStringTagShams() {
      return hasSymbols() && !!Symbol.toStringTag;
    };
  }
});

// node_modules/is-arguments/index.js
var require_is_arguments = __commonJS({
  "node_modules/is-arguments/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var hasToStringTag = require_shams2()();
    var callBound = require_callBound();
    var $toString = callBound("Object.prototype.toString");
    var isStandardArguments = function isArguments(value) {
      if (hasToStringTag && value && typeof value === "object" && Symbol.toStringTag in value) {
        return false;
      }
      return $toString(value) === "[object Arguments]";
    };
    var isLegacyArguments = function isArguments(value) {
      if (isStandardArguments(value)) {
        return true;
      }
      return value !== null && typeof value === "object" && typeof value.length === "number" && value.length >= 0 && $toString(value) !== "[object Array]" && $toString(value.callee) === "[object Function]";
    };
    var supportsStandardArguments = function() {
      return isStandardArguments(arguments);
    }();
    isStandardArguments.isLegacyArguments = isLegacyArguments;
    module.exports = supportsStandardArguments ? isStandardArguments : isLegacyArguments;
  }
});

// (disabled):node_modules/object-inspect/util.inspect
var require_util = __commonJS({
  "(disabled):node_modules/object-inspect/util.inspect"() {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
  }
});

// node_modules/object-inspect/index.js
var require_object_inspect = __commonJS({
  "node_modules/object-inspect/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var hasMap = typeof Map === "function" && Map.prototype;
    var mapSizeDescriptor = Object.getOwnPropertyDescriptor && hasMap ? Object.getOwnPropertyDescriptor(Map.prototype, "size") : null;
    var mapSize = hasMap && mapSizeDescriptor && typeof mapSizeDescriptor.get === "function" ? mapSizeDescriptor.get : null;
    var mapForEach = hasMap && Map.prototype.forEach;
    var hasSet = typeof Set === "function" && Set.prototype;
    var setSizeDescriptor = Object.getOwnPropertyDescriptor && hasSet ? Object.getOwnPropertyDescriptor(Set.prototype, "size") : null;
    var setSize = hasSet && setSizeDescriptor && typeof setSizeDescriptor.get === "function" ? setSizeDescriptor.get : null;
    var setForEach = hasSet && Set.prototype.forEach;
    var hasWeakMap = typeof WeakMap === "function" && WeakMap.prototype;
    var weakMapHas = hasWeakMap ? WeakMap.prototype.has : null;
    var hasWeakSet = typeof WeakSet === "function" && WeakSet.prototype;
    var weakSetHas = hasWeakSet ? WeakSet.prototype.has : null;
    var hasWeakRef = typeof WeakRef === "function" && WeakRef.prototype;
    var weakRefDeref = hasWeakRef ? WeakRef.prototype.deref : null;
    var booleanValueOf = Boolean.prototype.valueOf;
    var objectToString = Object.prototype.toString;
    var functionToString = Function.prototype.toString;
    var $match = String.prototype.match;
    var $slice = String.prototype.slice;
    var $replace = String.prototype.replace;
    var $toUpperCase = String.prototype.toUpperCase;
    var $toLowerCase = String.prototype.toLowerCase;
    var $test = RegExp.prototype.test;
    var $concat = Array.prototype.concat;
    var $join = Array.prototype.join;
    var $arrSlice = Array.prototype.slice;
    var $floor = Math.floor;
    var bigIntValueOf = typeof BigInt === "function" ? BigInt.prototype.valueOf : null;
    var gOPS = Object.getOwnPropertySymbols;
    var symToString = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? Symbol.prototype.toString : null;
    var hasShammedSymbols = typeof Symbol === "function" && typeof Symbol.iterator === "object";
    var toStringTag = typeof Symbol === "function" && Symbol.toStringTag && (typeof Symbol.toStringTag === hasShammedSymbols ? "object" : "symbol") ? Symbol.toStringTag : null;
    var isEnumerable = Object.prototype.propertyIsEnumerable;
    var gPO = (typeof Reflect === "function" ? Reflect.getPrototypeOf : Object.getPrototypeOf) || ([].__proto__ === Array.prototype ? function(O5) {
      return O5.__proto__;
    } : null);
    function addNumericSeparator(num, str) {
      if (num === Infinity || num === -Infinity || num !== num || num && num > -1e3 && num < 1e3 || $test.call(/e/, str)) {
        return str;
      }
      var sepRegex = /[0-9](?=(?:[0-9]{3})+(?![0-9]))/g;
      if (typeof num === "number") {
        var int = num < 0 ? -$floor(-num) : $floor(num);
        if (int !== num) {
          var intStr = String(int);
          var dec = $slice.call(str, intStr.length + 1);
          return $replace.call(intStr, sepRegex, "$&_") + "." + $replace.call($replace.call(dec, /([0-9]{3})/g, "$&_"), /_$/, "");
        }
      }
      return $replace.call(str, sepRegex, "$&_");
    }
    var utilInspect = require_util();
    var inspectCustom = utilInspect.custom;
    var inspectSymbol = isSymbol3(inspectCustom) ? inspectCustom : null;
    module.exports = function inspect_(obj, options, depth, seen) {
      var opts = options || {};
      if (has(opts, "quoteStyle") && (opts.quoteStyle !== "single" && opts.quoteStyle !== "double")) {
        throw new TypeError('option "quoteStyle" must be "single" or "double"');
      }
      if (has(opts, "maxStringLength") && (typeof opts.maxStringLength === "number" ? opts.maxStringLength < 0 && opts.maxStringLength !== Infinity : opts.maxStringLength !== null)) {
        throw new TypeError('option "maxStringLength", if provided, must be a positive integer, Infinity, or `null`');
      }
      var customInspect = has(opts, "customInspect") ? opts.customInspect : true;
      if (typeof customInspect !== "boolean" && customInspect !== "symbol") {
        throw new TypeError("option \"customInspect\", if provided, must be `true`, `false`, or `'symbol'`");
      }
      if (has(opts, "indent") && opts.indent !== null && opts.indent !== "	" && !(parseInt(opts.indent, 10) === opts.indent && opts.indent > 0)) {
        throw new TypeError('option "indent" must be "\\t", an integer > 0, or `null`');
      }
      if (has(opts, "numericSeparator") && typeof opts.numericSeparator !== "boolean") {
        throw new TypeError('option "numericSeparator", if provided, must be `true` or `false`');
      }
      var numericSeparator = opts.numericSeparator;
      if (typeof obj === "undefined") {
        return "undefined";
      }
      if (obj === null) {
        return "null";
      }
      if (typeof obj === "boolean") {
        return obj ? "true" : "false";
      }
      if (typeof obj === "string") {
        return inspectString(obj, opts);
      }
      if (typeof obj === "number") {
        if (obj === 0) {
          return Infinity / obj > 0 ? "0" : "-0";
        }
        var str = String(obj);
        return numericSeparator ? addNumericSeparator(obj, str) : str;
      }
      if (typeof obj === "bigint") {
        var bigIntStr = String(obj) + "n";
        return numericSeparator ? addNumericSeparator(obj, bigIntStr) : bigIntStr;
      }
      var maxDepth = typeof opts.depth === "undefined" ? 5 : opts.depth;
      if (typeof depth === "undefined") {
        depth = 0;
      }
      if (depth >= maxDepth && maxDepth > 0 && typeof obj === "object") {
        return isArray4(obj) ? "[Array]" : "[Object]";
      }
      var indent = getIndent(opts, depth);
      if (typeof seen === "undefined") {
        seen = [];
      } else if (indexOf(seen, obj) >= 0) {
        return "[Circular]";
      }
      function inspect3(value, from, noIndent) {
        if (from) {
          seen = $arrSlice.call(seen);
          seen.push(from);
        }
        if (noIndent) {
          var newOpts = {
            depth: opts.depth
          };
          if (has(opts, "quoteStyle")) {
            newOpts.quoteStyle = opts.quoteStyle;
          }
          return inspect_(value, newOpts, depth + 1, seen);
        }
        return inspect_(value, opts, depth + 1, seen);
      }
      if (typeof obj === "function" && !isRegExp4(obj)) {
        var name3 = nameOf(obj);
        var keys = arrObjKeys(obj, inspect3);
        return "[Function" + (name3 ? ": " + name3 : " (anonymous)") + "]" + (keys.length > 0 ? " { " + $join.call(keys, ", ") + " }" : "");
      }
      if (isSymbol3(obj)) {
        var symString = hasShammedSymbols ? $replace.call(String(obj), /^(Symbol\(.*\))_[^)]*$/, "$1") : symToString.call(obj);
        return typeof obj === "object" && !hasShammedSymbols ? markBoxed(symString) : symString;
      }
      if (isElement(obj)) {
        var s6 = "<" + $toLowerCase.call(String(obj.nodeName));
        var attrs = obj.attributes || [];
        for (var i7 = 0; i7 < attrs.length; i7++) {
          s6 += " " + attrs[i7].name + "=" + wrapQuotes(quote(attrs[i7].value), "double", opts);
        }
        s6 += ">";
        if (obj.childNodes && obj.childNodes.length) {
          s6 += "...";
        }
        s6 += "</" + $toLowerCase.call(String(obj.nodeName)) + ">";
        return s6;
      }
      if (isArray4(obj)) {
        if (obj.length === 0) {
          return "[]";
        }
        var xs = arrObjKeys(obj, inspect3);
        if (indent && !singleLineValues(xs)) {
          return "[" + indentedJoin(xs, indent) + "]";
        }
        return "[ " + $join.call(xs, ", ") + " ]";
      }
      if (isError3(obj)) {
        var parts = arrObjKeys(obj, inspect3);
        if (!("cause" in Error.prototype) && "cause" in obj && !isEnumerable.call(obj, "cause")) {
          return "{ [" + String(obj) + "] " + $join.call($concat.call("[cause]: " + inspect3(obj.cause), parts), ", ") + " }";
        }
        if (parts.length === 0) {
          return "[" + String(obj) + "]";
        }
        return "{ [" + String(obj) + "] " + $join.call(parts, ", ") + " }";
      }
      if (typeof obj === "object" && customInspect) {
        if (inspectSymbol && typeof obj[inspectSymbol] === "function" && utilInspect) {
          return utilInspect(obj, { depth: maxDepth - depth });
        } else if (customInspect !== "symbol" && typeof obj.inspect === "function") {
          return obj.inspect();
        }
      }
      if (isMap(obj)) {
        var mapParts = [];
        if (mapForEach) {
          mapForEach.call(obj, function(value, key) {
            mapParts.push(inspect3(key, obj, true) + " => " + inspect3(value, obj));
          });
        }
        return collectionOf("Map", mapSize.call(obj), mapParts, indent);
      }
      if (isSet(obj)) {
        var setParts = [];
        if (setForEach) {
          setForEach.call(obj, function(value) {
            setParts.push(inspect3(value, obj));
          });
        }
        return collectionOf("Set", setSize.call(obj), setParts, indent);
      }
      if (isWeakMap(obj)) {
        return weakCollectionOf("WeakMap");
      }
      if (isWeakSet(obj)) {
        return weakCollectionOf("WeakSet");
      }
      if (isWeakRef(obj)) {
        return weakCollectionOf("WeakRef");
      }
      if (isNumber4(obj)) {
        return markBoxed(inspect3(Number(obj)));
      }
      if (isBigInt(obj)) {
        return markBoxed(inspect3(bigIntValueOf.call(obj)));
      }
      if (isBoolean4(obj)) {
        return markBoxed(booleanValueOf.call(obj));
      }
      if (isString5(obj)) {
        return markBoxed(inspect3(String(obj)));
      }
      if (!isDate4(obj) && !isRegExp4(obj)) {
        var ys = arrObjKeys(obj, inspect3);
        var isPlainObject2 = gPO ? gPO(obj) === Object.prototype : obj instanceof Object || obj.constructor === Object;
        var protoTag = obj instanceof Object ? "" : "null prototype";
        var stringTag = !isPlainObject2 && toStringTag && Object(obj) === obj && toStringTag in obj ? $slice.call(toStr(obj), 8, -1) : protoTag ? "Object" : "";
        var constructorTag = isPlainObject2 || typeof obj.constructor !== "function" ? "" : obj.constructor.name ? obj.constructor.name + " " : "";
        var tag = constructorTag + (stringTag || protoTag ? "[" + $join.call($concat.call([], stringTag || [], protoTag || []), ": ") + "] " : "");
        if (ys.length === 0) {
          return tag + "{}";
        }
        if (indent) {
          return tag + "{" + indentedJoin(ys, indent) + "}";
        }
        return tag + "{ " + $join.call(ys, ", ") + " }";
      }
      return String(obj);
    };
    function wrapQuotes(s6, defaultStyle, opts) {
      var quoteChar = (opts.quoteStyle || defaultStyle) === "double" ? '"' : "'";
      return quoteChar + s6 + quoteChar;
    }
    function quote(s6) {
      return $replace.call(String(s6), /"/g, "&quot;");
    }
    function isArray4(obj) {
      return toStr(obj) === "[object Array]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isDate4(obj) {
      return toStr(obj) === "[object Date]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isRegExp4(obj) {
      return toStr(obj) === "[object RegExp]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isError3(obj) {
      return toStr(obj) === "[object Error]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isString5(obj) {
      return toStr(obj) === "[object String]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isNumber4(obj) {
      return toStr(obj) === "[object Number]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isBoolean4(obj) {
      return toStr(obj) === "[object Boolean]" && (!toStringTag || !(typeof obj === "object" && toStringTag in obj));
    }
    function isSymbol3(obj) {
      if (hasShammedSymbols) {
        return obj && typeof obj === "object" && obj instanceof Symbol;
      }
      if (typeof obj === "symbol") {
        return true;
      }
      if (!obj || typeof obj !== "object" || !symToString) {
        return false;
      }
      try {
        symToString.call(obj);
        return true;
      } catch (e10) {
      }
      return false;
    }
    function isBigInt(obj) {
      if (!obj || typeof obj !== "object" || !bigIntValueOf) {
        return false;
      }
      try {
        bigIntValueOf.call(obj);
        return true;
      } catch (e10) {
      }
      return false;
    }
    var hasOwn = Object.prototype.hasOwnProperty || function(key) {
      return key in this;
    };
    function has(obj, key) {
      return hasOwn.call(obj, key);
    }
    function toStr(obj) {
      return objectToString.call(obj);
    }
    function nameOf(f7) {
      if (f7.name) {
        return f7.name;
      }
      var m6 = $match.call(functionToString.call(f7), /^function\s*([\w$]+)/);
      if (m6) {
        return m6[1];
      }
      return null;
    }
    function indexOf(xs, x4) {
      if (xs.indexOf) {
        return xs.indexOf(x4);
      }
      for (var i7 = 0, l7 = xs.length; i7 < l7; i7++) {
        if (xs[i7] === x4) {
          return i7;
        }
      }
      return -1;
    }
    function isMap(x4) {
      if (!mapSize || !x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        mapSize.call(x4);
        try {
          setSize.call(x4);
        } catch (s6) {
          return true;
        }
        return x4 instanceof Map;
      } catch (e10) {
      }
      return false;
    }
    function isWeakMap(x4) {
      if (!weakMapHas || !x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        weakMapHas.call(x4, weakMapHas);
        try {
          weakSetHas.call(x4, weakSetHas);
        } catch (s6) {
          return true;
        }
        return x4 instanceof WeakMap;
      } catch (e10) {
      }
      return false;
    }
    function isWeakRef(x4) {
      if (!weakRefDeref || !x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        weakRefDeref.call(x4);
        return true;
      } catch (e10) {
      }
      return false;
    }
    function isSet(x4) {
      if (!setSize || !x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        setSize.call(x4);
        try {
          mapSize.call(x4);
        } catch (m6) {
          return true;
        }
        return x4 instanceof Set;
      } catch (e10) {
      }
      return false;
    }
    function isWeakSet(x4) {
      if (!weakSetHas || !x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        weakSetHas.call(x4, weakSetHas);
        try {
          weakMapHas.call(x4, weakMapHas);
        } catch (s6) {
          return true;
        }
        return x4 instanceof WeakSet;
      } catch (e10) {
      }
      return false;
    }
    function isElement(x4) {
      if (!x4 || typeof x4 !== "object") {
        return false;
      }
      if (typeof HTMLElement !== "undefined" && x4 instanceof HTMLElement) {
        return true;
      }
      return typeof x4.nodeName === "string" && typeof x4.getAttribute === "function";
    }
    function inspectString(str, opts) {
      if (str.length > opts.maxStringLength) {
        var remaining = str.length - opts.maxStringLength;
        var trailer = "... " + remaining + " more character" + (remaining > 1 ? "s" : "");
        return inspectString($slice.call(str, 0, opts.maxStringLength), opts) + trailer;
      }
      var s6 = $replace.call($replace.call(str, /(['\\])/g, "\\$1"), /[\x00-\x1f]/g, lowbyte);
      return wrapQuotes(s6, "single", opts);
    }
    function lowbyte(c7) {
      var n9 = c7.charCodeAt(0);
      var x4 = {
        8: "b",
        9: "t",
        10: "n",
        12: "f",
        13: "r"
      }[n9];
      if (x4) {
        return "\\" + x4;
      }
      return "\\x" + (n9 < 16 ? "0" : "") + $toUpperCase.call(n9.toString(16));
    }
    function markBoxed(str) {
      return "Object(" + str + ")";
    }
    function weakCollectionOf(type2) {
      return type2 + " { ? }";
    }
    function collectionOf(type2, size, entries, indent) {
      var joinedEntries = indent ? indentedJoin(entries, indent) : $join.call(entries, ", ");
      return type2 + " (" + size + ") {" + joinedEntries + "}";
    }
    function singleLineValues(xs) {
      for (var i7 = 0; i7 < xs.length; i7++) {
        if (indexOf(xs[i7], "\n") >= 0) {
          return false;
        }
      }
      return true;
    }
    function getIndent(opts, depth) {
      var baseIndent;
      if (opts.indent === "	") {
        baseIndent = "	";
      } else if (typeof opts.indent === "number" && opts.indent > 0) {
        baseIndent = $join.call(Array(opts.indent + 1), " ");
      } else {
        return null;
      }
      return {
        base: baseIndent,
        prev: $join.call(Array(depth + 1), baseIndent)
      };
    }
    function indentedJoin(xs, indent) {
      if (xs.length === 0) {
        return "";
      }
      var lineJoiner = "\n" + indent.prev + indent.base;
      return lineJoiner + $join.call(xs, "," + lineJoiner) + "\n" + indent.prev;
    }
    function arrObjKeys(obj, inspect3) {
      var isArr = isArray4(obj);
      var xs = [];
      if (isArr) {
        xs.length = obj.length;
        for (var i7 = 0; i7 < obj.length; i7++) {
          xs[i7] = has(obj, i7) ? inspect3(obj[i7], obj) : "";
        }
      }
      var syms = typeof gOPS === "function" ? gOPS(obj) : [];
      var symMap;
      if (hasShammedSymbols) {
        symMap = {};
        for (var k4 = 0; k4 < syms.length; k4++) {
          symMap["$" + syms[k4]] = syms[k4];
        }
      }
      for (var key in obj) {
        if (!has(obj, key)) {
          continue;
        }
        if (isArr && String(Number(key)) === key && key < obj.length) {
          continue;
        }
        if (hasShammedSymbols && symMap["$" + key] instanceof Symbol) {
          continue;
        } else if ($test.call(/[^\w$]/, key)) {
          xs.push(inspect3(key, obj) + ": " + inspect3(obj[key], obj));
        } else {
          xs.push(key + ": " + inspect3(obj[key], obj));
        }
      }
      if (typeof gOPS === "function") {
        for (var j4 = 0; j4 < syms.length; j4++) {
          if (isEnumerable.call(obj, syms[j4])) {
            xs.push("[" + inspect3(syms[j4]) + "]: " + inspect3(obj[syms[j4]], obj));
          }
        }
      }
      return xs;
    }
  }
});

// node_modules/side-channel/index.js
var require_side_channel = __commonJS({
  "node_modules/side-channel/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var callBound = require_callBound();
    var inspect3 = require_object_inspect();
    var $TypeError = GetIntrinsic("%TypeError%");
    var $WeakMap = GetIntrinsic("%WeakMap%", true);
    var $Map = GetIntrinsic("%Map%", true);
    var $weakMapGet = callBound("WeakMap.prototype.get", true);
    var $weakMapSet = callBound("WeakMap.prototype.set", true);
    var $weakMapHas = callBound("WeakMap.prototype.has", true);
    var $mapGet = callBound("Map.prototype.get", true);
    var $mapSet = callBound("Map.prototype.set", true);
    var $mapHas = callBound("Map.prototype.has", true);
    var listGetNode = function(list, key) {
      for (var prev = list, curr; (curr = prev.next) !== null; prev = curr) {
        if (curr.key === key) {
          prev.next = curr.next;
          curr.next = list.next;
          list.next = curr;
          return curr;
        }
      }
    };
    var listGet = function(objects, key) {
      var node = listGetNode(objects, key);
      return node && node.value;
    };
    var listSet = function(objects, key, value) {
      var node = listGetNode(objects, key);
      if (node) {
        node.value = value;
      } else {
        objects.next = {
          // eslint-disable-line no-param-reassign
          key,
          next: objects.next,
          value
        };
      }
    };
    var listHas = function(objects, key) {
      return !!listGetNode(objects, key);
    };
    module.exports = function getSideChannel() {
      var $wm;
      var $m;
      var $o;
      var channel = {
        assert: function(key) {
          if (!channel.has(key)) {
            throw new $TypeError("Side channel does not contain " + inspect3(key));
          }
        },
        get: function(key) {
          if ($WeakMap && key && (typeof key === "object" || typeof key === "function")) {
            if ($wm) {
              return $weakMapGet($wm, key);
            }
          } else if ($Map) {
            if ($m) {
              return $mapGet($m, key);
            }
          } else {
            if ($o) {
              return listGet($o, key);
            }
          }
        },
        has: function(key) {
          if ($WeakMap && key && (typeof key === "object" || typeof key === "function")) {
            if ($wm) {
              return $weakMapHas($wm, key);
            }
          } else if ($Map) {
            if ($m) {
              return $mapHas($m, key);
            }
          } else {
            if ($o) {
              return listHas($o, key);
            }
          }
          return false;
        },
        set: function(key, value) {
          if ($WeakMap && key && (typeof key === "object" || typeof key === "function")) {
            if (!$wm) {
              $wm = new $WeakMap();
            }
            $weakMapSet($wm, key, value);
          } else if ($Map) {
            if (!$m) {
              $m = new $Map();
            }
            $mapSet($m, key, value);
          } else {
            if (!$o) {
              $o = { key: {}, next: null };
            }
            listSet($o, key, value);
          }
        }
      };
      return channel;
    };
  }
});

// node_modules/internal-slot/index.js
var require_internal_slot = __commonJS({
  "node_modules/internal-slot/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var has = require_src();
    var channel = require_side_channel()();
    var $TypeError = GetIntrinsic("%TypeError%");
    var SLOT = {
      assert: function(O5, slot) {
        if (!O5 || typeof O5 !== "object" && typeof O5 !== "function") {
          throw new $TypeError("`O` is not an object");
        }
        if (typeof slot !== "string") {
          throw new $TypeError("`slot` must be a string");
        }
        channel.assert(O5);
        if (!SLOT.has(O5, slot)) {
          throw new $TypeError("`" + slot + "` is not present on `O`");
        }
      },
      get: function(O5, slot) {
        if (!O5 || typeof O5 !== "object" && typeof O5 !== "function") {
          throw new $TypeError("`O` is not an object");
        }
        if (typeof slot !== "string") {
          throw new $TypeError("`slot` must be a string");
        }
        var slots = channel.get(O5);
        return slots && slots["$" + slot];
      },
      has: function(O5, slot) {
        if (!O5 || typeof O5 !== "object" && typeof O5 !== "function") {
          throw new $TypeError("`O` is not an object");
        }
        if (typeof slot !== "string") {
          throw new $TypeError("`slot` must be a string");
        }
        var slots = channel.get(O5);
        return !!slots && has(slots, "$" + slot);
      },
      set: function(O5, slot, V3) {
        if (!O5 || typeof O5 !== "object" && typeof O5 !== "function") {
          throw new $TypeError("`O` is not an object");
        }
        if (typeof slot !== "string") {
          throw new $TypeError("`slot` must be a string");
        }
        var slots = channel.get(O5);
        if (!slots) {
          slots = {};
          channel.set(O5, slots);
        }
        slots["$" + slot] = V3;
      }
    };
    if (Object.freeze) {
      Object.freeze(SLOT);
    }
    module.exports = SLOT;
  }
});

// node_modules/stop-iteration-iterator/index.js
var require_stop_iteration_iterator = __commonJS({
  "node_modules/stop-iteration-iterator/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var SLOT = require_internal_slot();
    var $SyntaxError = SyntaxError;
    var $StopIteration = typeof StopIteration === "object" ? StopIteration : null;
    module.exports = function getStopIterationIterator(origIterator) {
      if (!$StopIteration) {
        throw new $SyntaxError("this environment lacks StopIteration");
      }
      SLOT.set(origIterator, "[[Done]]", false);
      var siIterator = {
        next: function next() {
          var iterator = SLOT.get(this, "[[Iterator]]");
          var done = SLOT.get(iterator, "[[Done]]");
          try {
            return {
              done,
              value: done ? void 0 : iterator.next()
            };
          } catch (e10) {
            SLOT.set(iterator, "[[Done]]", true);
            if (e10 !== $StopIteration) {
              throw e10;
            }
            return {
              done: true,
              value: void 0
            };
          }
        }
      };
      SLOT.set(siIterator, "[[Iterator]]", origIterator);
      return siIterator;
    };
  }
});

// node_modules/isarray/index.js
var require_isarray = __commonJS({
  "node_modules/isarray/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var toString3 = {}.toString;
    module.exports = Array.isArray || function(arr) {
      return toString3.call(arr) == "[object Array]";
    };
  }
});

// node_modules/is-string/index.js
var require_is_string = __commonJS({
  "node_modules/is-string/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var strValue = String.prototype.valueOf;
    var tryStringObject = function tryStringObject2(value) {
      try {
        strValue.call(value);
        return true;
      } catch (e10) {
        return false;
      }
    };
    var toStr = Object.prototype.toString;
    var strClass = "[object String]";
    var hasToStringTag = require_shams2()();
    module.exports = function isString5(value) {
      if (typeof value === "string") {
        return true;
      }
      if (typeof value !== "object") {
        return false;
      }
      return hasToStringTag ? tryStringObject(value) : toStr.call(value) === strClass;
    };
  }
});

// node_modules/is-map/index.js
var require_is_map = __commonJS({
  "node_modules/is-map/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var $Map = typeof Map === "function" && Map.prototype ? Map : null;
    var $Set = typeof Set === "function" && Set.prototype ? Set : null;
    var exported;
    if (!$Map) {
      exported = function isMap(x4) {
        return false;
      };
    }
    var $mapHas = $Map ? Map.prototype.has : null;
    var $setHas = $Set ? Set.prototype.has : null;
    if (!exported && !$mapHas) {
      exported = function isMap(x4) {
        return false;
      };
    }
    module.exports = exported || function isMap(x4) {
      if (!x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        $mapHas.call(x4);
        if ($setHas) {
          try {
            $setHas.call(x4);
          } catch (e10) {
            return true;
          }
        }
        return x4 instanceof $Map;
      } catch (e10) {
      }
      return false;
    };
  }
});

// node_modules/is-set/index.js
var require_is_set = __commonJS({
  "node_modules/is-set/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var $Map = typeof Map === "function" && Map.prototype ? Map : null;
    var $Set = typeof Set === "function" && Set.prototype ? Set : null;
    var exported;
    if (!$Set) {
      exported = function isSet(x4) {
        return false;
      };
    }
    var $mapHas = $Map ? Map.prototype.has : null;
    var $setHas = $Set ? Set.prototype.has : null;
    if (!exported && !$setHas) {
      exported = function isSet(x4) {
        return false;
      };
    }
    module.exports = exported || function isSet(x4) {
      if (!x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        $setHas.call(x4);
        if ($mapHas) {
          try {
            $mapHas.call(x4);
          } catch (e10) {
            return true;
          }
        }
        return x4 instanceof $Set;
      } catch (e10) {
      }
      return false;
    };
  }
});

// node_modules/es-get-iterator/index.js
var require_es_get_iterator = __commonJS({
  "node_modules/es-get-iterator/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var isArguments = require_is_arguments();
    var getStopIterationIterator = require_stop_iteration_iterator();
    if (require_has_symbols()() || require_shams()()) {
      $iterator = Symbol.iterator;
      module.exports = function getIterator(iterable) {
        if (iterable != null && typeof iterable[$iterator] !== "undefined") {
          return iterable[$iterator]();
        }
        if (isArguments(iterable)) {
          return Array.prototype[$iterator].call(iterable);
        }
      };
    } else {
      isArray4 = require_isarray();
      isString5 = require_is_string();
      GetIntrinsic = require_get_intrinsic();
      $Map = GetIntrinsic("%Map%", true);
      $Set = GetIntrinsic("%Set%", true);
      callBound = require_callBound();
      $arrayPush = callBound("Array.prototype.push");
      $charCodeAt = callBound("String.prototype.charCodeAt");
      $stringSlice = callBound("String.prototype.slice");
      advanceStringIndex = function advanceStringIndex2(S4, index) {
        var length = S4.length;
        if (index + 1 >= length) {
          return index + 1;
        }
        var first = $charCodeAt(S4, index);
        if (first < 55296 || first > 56319) {
          return index + 1;
        }
        var second = $charCodeAt(S4, index + 1);
        if (second < 56320 || second > 57343) {
          return index + 1;
        }
        return index + 2;
      };
      getArrayIterator = function getArrayIterator2(arraylike) {
        var i7 = 0;
        return {
          next: function next() {
            var done = i7 >= arraylike.length;
            var value;
            if (!done) {
              value = arraylike[i7];
              i7 += 1;
            }
            return {
              done,
              value
            };
          }
        };
      };
      getNonCollectionIterator = function getNonCollectionIterator2(iterable, noPrimordialCollections) {
        if (isArray4(iterable) || isArguments(iterable)) {
          return getArrayIterator(iterable);
        }
        if (isString5(iterable)) {
          var i7 = 0;
          return {
            next: function next() {
              var nextIndex = advanceStringIndex(iterable, i7);
              var value = $stringSlice(iterable, i7, nextIndex);
              i7 = nextIndex;
              return {
                done: nextIndex > iterable.length,
                value
              };
            }
          };
        }
        if (noPrimordialCollections && typeof iterable["_es6-shim iterator_"] !== "undefined") {
          return iterable["_es6-shim iterator_"]();
        }
      };
      if (!$Map && !$Set) {
        module.exports = function getIterator(iterable) {
          if (iterable != null) {
            return getNonCollectionIterator(iterable, true);
          }
        };
      } else {
        isMap = require_is_map();
        isSet = require_is_set();
        $mapForEach = callBound("Map.prototype.forEach", true);
        $setForEach = callBound("Set.prototype.forEach", true);
        if (typeof process_exports === "undefined" || !process_exports.versions || !process_exports.versions.node) {
          $mapIterator = callBound("Map.prototype.iterator", true);
          $setIterator = callBound("Set.prototype.iterator", true);
        }
        $mapAtAtIterator = callBound("Map.prototype.@@iterator", true) || callBound("Map.prototype._es6-shim iterator_", true);
        $setAtAtIterator = callBound("Set.prototype.@@iterator", true) || callBound("Set.prototype._es6-shim iterator_", true);
        getCollectionIterator = function getCollectionIterator2(iterable) {
          if (isMap(iterable)) {
            if ($mapIterator) {
              return getStopIterationIterator($mapIterator(iterable));
            }
            if ($mapAtAtIterator) {
              return $mapAtAtIterator(iterable);
            }
            if ($mapForEach) {
              var entries = [];
              $mapForEach(iterable, function(v7, k4) {
                $arrayPush(entries, [k4, v7]);
              });
              return getArrayIterator(entries);
            }
          }
          if (isSet(iterable)) {
            if ($setIterator) {
              return getStopIterationIterator($setIterator(iterable));
            }
            if ($setAtAtIterator) {
              return $setAtAtIterator(iterable);
            }
            if ($setForEach) {
              var values = [];
              $setForEach(iterable, function(v7) {
                $arrayPush(values, v7);
              });
              return getArrayIterator(values);
            }
          }
        };
        module.exports = function getIterator(iterable) {
          return getCollectionIterator(iterable) || getNonCollectionIterator(iterable);
        };
      }
    }
    var $iterator;
    var isArray4;
    var isString5;
    var GetIntrinsic;
    var $Map;
    var $Set;
    var callBound;
    var $arrayPush;
    var $charCodeAt;
    var $stringSlice;
    var advanceStringIndex;
    var getArrayIterator;
    var getNonCollectionIterator;
    var isMap;
    var isSet;
    var $mapForEach;
    var $setForEach;
    var $mapIterator;
    var $setIterator;
    var $mapAtAtIterator;
    var $setAtAtIterator;
    var getCollectionIterator;
  }
});

// node_modules/object-is/implementation.js
var require_implementation5 = __commonJS({
  "node_modules/object-is/implementation.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var numberIsNaN = function(value) {
      return value !== value;
    };
    module.exports = function is(a7, b5) {
      if (a7 === 0 && b5 === 0) {
        return 1 / a7 === 1 / b5;
      }
      if (a7 === b5) {
        return true;
      }
      if (numberIsNaN(a7) && numberIsNaN(b5)) {
        return true;
      }
      return false;
    };
  }
});

// node_modules/object-is/polyfill.js
var require_polyfill3 = __commonJS({
  "node_modules/object-is/polyfill.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var implementation = require_implementation5();
    module.exports = function getPolyfill() {
      return typeof Object.is === "function" ? Object.is : implementation;
    };
  }
});

// node_modules/object-is/shim.js
var require_shim3 = __commonJS({
  "node_modules/object-is/shim.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var getPolyfill = require_polyfill3();
    var define2 = require_define_properties();
    module.exports = function shimObjectIs() {
      var polyfill = getPolyfill();
      define2(Object, { is: polyfill }, {
        is: function testObjectIs() {
          return Object.is !== polyfill;
        }
      });
      return polyfill;
    };
  }
});

// node_modules/object-is/index.js
var require_object_is = __commonJS({
  "node_modules/object-is/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var define2 = require_define_properties();
    var callBind = require_call_bind();
    var implementation = require_implementation5();
    var getPolyfill = require_polyfill3();
    var shim = require_shim3();
    var polyfill = callBind(getPolyfill(), Object);
    define2(polyfill, {
      getPolyfill,
      implementation,
      shim
    });
    module.exports = polyfill;
  }
});

// node_modules/is-callable/index.js
var require_is_callable = __commonJS({
  "node_modules/is-callable/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var fnToStr = Function.prototype.toString;
    var reflectApply = typeof Reflect === "object" && Reflect !== null && Reflect.apply;
    var badArrayLike;
    var isCallableMarker;
    if (typeof reflectApply === "function" && typeof Object.defineProperty === "function") {
      try {
        badArrayLike = Object.defineProperty({}, "length", {
          get: function() {
            throw isCallableMarker;
          }
        });
        isCallableMarker = {};
        reflectApply(function() {
          throw 42;
        }, null, badArrayLike);
      } catch (_4) {
        if (_4 !== isCallableMarker) {
          reflectApply = null;
        }
      }
    } else {
      reflectApply = null;
    }
    var constructorRegex = /^\s*class\b/;
    var isES6ClassFn = function isES6ClassFunction(value) {
      try {
        var fnStr = fnToStr.call(value);
        return constructorRegex.test(fnStr);
      } catch (e10) {
        return false;
      }
    };
    var tryFunctionObject = function tryFunctionToStr(value) {
      try {
        if (isES6ClassFn(value)) {
          return false;
        }
        fnToStr.call(value);
        return true;
      } catch (e10) {
        return false;
      }
    };
    var toStr = Object.prototype.toString;
    var objectClass = "[object Object]";
    var fnClass = "[object Function]";
    var genClass = "[object GeneratorFunction]";
    var ddaClass = "[object HTMLAllCollection]";
    var ddaClass2 = "[object HTML document.all class]";
    var ddaClass3 = "[object HTMLCollection]";
    var hasToStringTag = typeof Symbol === "function" && !!Symbol.toStringTag;
    var isIE68 = !(0 in [,]);
    var isDDA = function isDocumentDotAll() {
      return false;
    };
    if (typeof document === "object") {
      all3 = document.all;
      if (toStr.call(all3) === toStr.call(document.all)) {
        isDDA = function isDocumentDotAll(value) {
          if ((isIE68 || !value) && (typeof value === "undefined" || typeof value === "object")) {
            try {
              var str = toStr.call(value);
              return (str === ddaClass || str === ddaClass2 || str === ddaClass3 || str === objectClass) && value("") == null;
            } catch (e10) {
            }
          }
          return false;
        };
      }
    }
    var all3;
    module.exports = reflectApply ? function isCallable(value) {
      if (isDDA(value)) {
        return true;
      }
      if (!value) {
        return false;
      }
      if (typeof value !== "function" && typeof value !== "object") {
        return false;
      }
      try {
        reflectApply(value, null, badArrayLike);
      } catch (e10) {
        if (e10 !== isCallableMarker) {
          return false;
        }
      }
      return !isES6ClassFn(value) && tryFunctionObject(value);
    } : function isCallable(value) {
      if (isDDA(value)) {
        return true;
      }
      if (!value) {
        return false;
      }
      if (typeof value !== "function" && typeof value !== "object") {
        return false;
      }
      if (hasToStringTag) {
        return tryFunctionObject(value);
      }
      if (isES6ClassFn(value)) {
        return false;
      }
      var strClass = toStr.call(value);
      if (strClass !== fnClass && strClass !== genClass && !/^\[object HTML/.test(strClass)) {
        return false;
      }
      return tryFunctionObject(value);
    };
  }
});

// node_modules/for-each/index.js
var require_for_each = __commonJS({
  "node_modules/for-each/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var isCallable = require_is_callable();
    var toStr = Object.prototype.toString;
    var hasOwnProperty2 = Object.prototype.hasOwnProperty;
    var forEachArray = function forEachArray2(array, iterator, receiver) {
      for (var i7 = 0, len = array.length; i7 < len; i7++) {
        if (hasOwnProperty2.call(array, i7)) {
          if (receiver == null) {
            iterator(array[i7], i7, array);
          } else {
            iterator.call(receiver, array[i7], i7, array);
          }
        }
      }
    };
    var forEachString = function forEachString2(string, iterator, receiver) {
      for (var i7 = 0, len = string.length; i7 < len; i7++) {
        if (receiver == null) {
          iterator(string.charAt(i7), i7, string);
        } else {
          iterator.call(receiver, string.charAt(i7), i7, string);
        }
      }
    };
    var forEachObject = function forEachObject2(object, iterator, receiver) {
      for (var k4 in object) {
        if (hasOwnProperty2.call(object, k4)) {
          if (receiver == null) {
            iterator(object[k4], k4, object);
          } else {
            iterator.call(receiver, object[k4], k4, object);
          }
        }
      }
    };
    var forEach2 = function forEach3(list, iterator, thisArg) {
      if (!isCallable(iterator)) {
        throw new TypeError("iterator must be a function");
      }
      var receiver;
      if (arguments.length >= 3) {
        receiver = thisArg;
      }
      if (toStr.call(list) === "[object Array]") {
        forEachArray(list, iterator, receiver);
      } else if (typeof list === "string") {
        forEachString(list, iterator, receiver);
      } else {
        forEachObject(list, iterator, receiver);
      }
    };
    module.exports = forEach2;
  }
});

// node_modules/available-typed-arrays/index.js
var require_available_typed_arrays = __commonJS({
  "node_modules/available-typed-arrays/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var possibleNames = [
      "BigInt64Array",
      "BigUint64Array",
      "Float32Array",
      "Float64Array",
      "Int16Array",
      "Int32Array",
      "Int8Array",
      "Uint16Array",
      "Uint32Array",
      "Uint8Array",
      "Uint8ClampedArray"
    ];
    var g5 = typeof globalThis === "undefined" ? global : globalThis;
    module.exports = function availableTypedArrays() {
      var out = [];
      for (var i7 = 0; i7 < possibleNames.length; i7++) {
        if (typeof g5[possibleNames[i7]] === "function") {
          out[out.length] = possibleNames[i7];
        }
      }
      return out;
    };
  }
});

// node_modules/gopd/index.js
var require_gopd = __commonJS({
  "node_modules/gopd/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var $gOPD = GetIntrinsic("%Object.getOwnPropertyDescriptor%", true);
    if ($gOPD) {
      try {
        $gOPD([], "length");
      } catch (e10) {
        $gOPD = null;
      }
    }
    module.exports = $gOPD;
  }
});

// node_modules/is-typed-array/index.js
var require_is_typed_array = __commonJS({
  "node_modules/is-typed-array/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var forEach2 = require_for_each();
    var availableTypedArrays = require_available_typed_arrays();
    var callBound = require_callBound();
    var $toString = callBound("Object.prototype.toString");
    var hasToStringTag = require_shams2()();
    var gOPD = require_gopd();
    var g5 = typeof globalThis === "undefined" ? global : globalThis;
    var typedArrays = availableTypedArrays();
    var $indexOf = callBound("Array.prototype.indexOf", true) || function indexOf(array, value) {
      for (var i7 = 0; i7 < array.length; i7 += 1) {
        if (array[i7] === value) {
          return i7;
        }
      }
      return -1;
    };
    var $slice = callBound("String.prototype.slice");
    var toStrTags = {};
    var getPrototypeOf2 = Object.getPrototypeOf;
    if (hasToStringTag && gOPD && getPrototypeOf2) {
      forEach2(typedArrays, function(typedArray) {
        var arr = new g5[typedArray]();
        if (Symbol.toStringTag in arr) {
          var proto = getPrototypeOf2(arr);
          var descriptor = gOPD(proto, Symbol.toStringTag);
          if (!descriptor) {
            var superProto = getPrototypeOf2(proto);
            descriptor = gOPD(superProto, Symbol.toStringTag);
          }
          toStrTags[typedArray] = descriptor.get;
        }
      });
    }
    var tryTypedArrays = function tryAllTypedArrays(value) {
      var anyTrue = false;
      forEach2(toStrTags, function(getter, typedArray) {
        if (!anyTrue) {
          try {
            anyTrue = getter.call(value) === typedArray;
          } catch (e10) {
          }
        }
      });
      return anyTrue;
    };
    module.exports = function isTypedArray2(value) {
      if (!value || typeof value !== "object") {
        return false;
      }
      if (!hasToStringTag || !(Symbol.toStringTag in value)) {
        var tag = $slice($toString(value), 8, -1);
        return $indexOf(typedArrays, tag) > -1;
      }
      if (!gOPD) {
        return false;
      }
      return tryTypedArrays(value);
    };
  }
});

// node_modules/is-array-buffer/index.js
var require_is_array_buffer = __commonJS({
  "node_modules/is-array-buffer/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var callBind = require_call_bind();
    var callBound = require_callBound();
    var GetIntrinsic = require_get_intrinsic();
    var isTypedArray2 = require_is_typed_array();
    var $ArrayBuffer = GetIntrinsic("ArrayBuffer", true);
    var $Float32Array = GetIntrinsic("Float32Array", true);
    var $byteLength = callBound("ArrayBuffer.prototype.byteLength", true);
    var abSlice = $ArrayBuffer && !$byteLength && new $ArrayBuffer().slice;
    var $abSlice = abSlice && callBind(abSlice);
    module.exports = $byteLength || $abSlice ? function isArrayBuffer2(obj) {
      if (!obj || typeof obj !== "object") {
        return false;
      }
      try {
        if ($byteLength) {
          $byteLength(obj);
        } else {
          $abSlice(obj, 0);
        }
        return true;
      } catch (e10) {
        return false;
      }
    } : $Float32Array ? function IsArrayBuffer(obj) {
      try {
        return new $Float32Array(obj).buffer === obj && !isTypedArray2(obj);
      } catch (e10) {
        return typeof obj === "object" && e10.name === "RangeError";
      }
    } : function isArrayBuffer2(obj) {
      return false;
    };
  }
});

// node_modules/is-date-object/index.js
var require_is_date_object = __commonJS({
  "node_modules/is-date-object/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var getDay = Date.prototype.getDay;
    var tryDateObject = function tryDateGetDayCall(value) {
      try {
        getDay.call(value);
        return true;
      } catch (e10) {
        return false;
      }
    };
    var toStr = Object.prototype.toString;
    var dateClass = "[object Date]";
    var hasToStringTag = require_shams2()();
    module.exports = function isDateObject(value) {
      if (typeof value !== "object" || value === null) {
        return false;
      }
      return hasToStringTag ? tryDateObject(value) : toStr.call(value) === dateClass;
    };
  }
});

// node_modules/is-regex/index.js
var require_is_regex = __commonJS({
  "node_modules/is-regex/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var callBound = require_callBound();
    var hasToStringTag = require_shams2()();
    var has;
    var $exec;
    var isRegexMarker;
    var badStringifier;
    if (hasToStringTag) {
      has = callBound("Object.prototype.hasOwnProperty");
      $exec = callBound("RegExp.prototype.exec");
      isRegexMarker = {};
      throwRegexMarker = function() {
        throw isRegexMarker;
      };
      badStringifier = {
        toString: throwRegexMarker,
        valueOf: throwRegexMarker
      };
      if (typeof Symbol.toPrimitive === "symbol") {
        badStringifier[Symbol.toPrimitive] = throwRegexMarker;
      }
    }
    var throwRegexMarker;
    var $toString = callBound("Object.prototype.toString");
    var gOPD = Object.getOwnPropertyDescriptor;
    var regexClass = "[object RegExp]";
    module.exports = hasToStringTag ? function isRegex(value) {
      if (!value || typeof value !== "object") {
        return false;
      }
      var descriptor = gOPD(value, "lastIndex");
      var hasLastIndexDataProperty = descriptor && has(descriptor, "value");
      if (!hasLastIndexDataProperty) {
        return false;
      }
      try {
        $exec(value, badStringifier);
      } catch (e10) {
        return e10 === isRegexMarker;
      }
    } : function isRegex(value) {
      if (!value || typeof value !== "object" && typeof value !== "function") {
        return false;
      }
      return $toString(value) === regexClass;
    };
  }
});

// node_modules/is-shared-array-buffer/index.js
var require_is_shared_array_buffer = __commonJS({
  "node_modules/is-shared-array-buffer/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var callBound = require_callBound();
    var $byteLength = callBound("SharedArrayBuffer.prototype.byteLength", true);
    module.exports = $byteLength ? function isSharedArrayBuffer(obj) {
      if (!obj || typeof obj !== "object") {
        return false;
      }
      try {
        $byteLength(obj);
        return true;
      } catch (e10) {
        return false;
      }
    } : function isSharedArrayBuffer(obj) {
      return false;
    };
  }
});

// node_modules/is-number-object/index.js
var require_is_number_object = __commonJS({
  "node_modules/is-number-object/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var numToStr = Number.prototype.toString;
    var tryNumberObject = function tryNumberObject2(value) {
      try {
        numToStr.call(value);
        return true;
      } catch (e10) {
        return false;
      }
    };
    var toStr = Object.prototype.toString;
    var numClass = "[object Number]";
    var hasToStringTag = require_shams2()();
    module.exports = function isNumberObject(value) {
      if (typeof value === "number") {
        return true;
      }
      if (typeof value !== "object") {
        return false;
      }
      return hasToStringTag ? tryNumberObject(value) : toStr.call(value) === numClass;
    };
  }
});

// node_modules/is-boolean-object/index.js
var require_is_boolean_object = __commonJS({
  "node_modules/is-boolean-object/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var callBound = require_callBound();
    var $boolToStr = callBound("Boolean.prototype.toString");
    var $toString = callBound("Object.prototype.toString");
    var tryBooleanObject = function booleanBrandCheck(value) {
      try {
        $boolToStr(value);
        return true;
      } catch (e10) {
        return false;
      }
    };
    var boolClass = "[object Boolean]";
    var hasToStringTag = require_shams2()();
    module.exports = function isBoolean4(value) {
      if (typeof value === "boolean") {
        return true;
      }
      if (value === null || typeof value !== "object") {
        return false;
      }
      return hasToStringTag && Symbol.toStringTag in value ? tryBooleanObject(value) : $toString(value) === boolClass;
    };
  }
});

// node_modules/is-symbol/index.js
var require_is_symbol = __commonJS({
  "node_modules/is-symbol/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var toStr = Object.prototype.toString;
    var hasSymbols = require_has_symbols()();
    if (hasSymbols) {
      symToStr = Symbol.prototype.toString;
      symStringRegex = /^Symbol\(.*\)$/;
      isSymbolObject = function isRealSymbolObject(value) {
        if (typeof value.valueOf() !== "symbol") {
          return false;
        }
        return symStringRegex.test(symToStr.call(value));
      };
      module.exports = function isSymbol3(value) {
        if (typeof value === "symbol") {
          return true;
        }
        if (toStr.call(value) !== "[object Symbol]") {
          return false;
        }
        try {
          return isSymbolObject(value);
        } catch (e10) {
          return false;
        }
      };
    } else {
      module.exports = function isSymbol3(value) {
        return false;
      };
    }
    var symToStr;
    var symStringRegex;
    var isSymbolObject;
  }
});

// node_modules/has-bigints/index.js
var require_has_bigints = __commonJS({
  "node_modules/has-bigints/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var $BigInt = typeof BigInt !== "undefined" && BigInt;
    module.exports = function hasNativeBigInts() {
      return typeof $BigInt === "function" && typeof BigInt === "function" && typeof $BigInt(42) === "bigint" && typeof BigInt(42) === "bigint";
    };
  }
});

// node_modules/is-bigint/index.js
var require_is_bigint = __commonJS({
  "node_modules/is-bigint/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var hasBigInts = require_has_bigints()();
    if (hasBigInts) {
      bigIntValueOf = BigInt.prototype.valueOf;
      tryBigInt = function tryBigIntObject(value) {
        try {
          bigIntValueOf.call(value);
          return true;
        } catch (e10) {
        }
        return false;
      };
      module.exports = function isBigInt(value) {
        if (value === null || typeof value === "undefined" || typeof value === "boolean" || typeof value === "string" || typeof value === "number" || typeof value === "symbol" || typeof value === "function") {
          return false;
        }
        if (typeof value === "bigint") {
          return true;
        }
        return tryBigInt(value);
      };
    } else {
      module.exports = function isBigInt(value) {
        return false;
      };
    }
    var bigIntValueOf;
    var tryBigInt;
  }
});

// node_modules/which-boxed-primitive/index.js
var require_which_boxed_primitive = __commonJS({
  "node_modules/which-boxed-primitive/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var isString5 = require_is_string();
    var isNumber4 = require_is_number_object();
    var isBoolean4 = require_is_boolean_object();
    var isSymbol3 = require_is_symbol();
    var isBigInt = require_is_bigint();
    module.exports = function whichBoxedPrimitive(value) {
      if (value == null || typeof value !== "object" && typeof value !== "function") {
        return null;
      }
      if (isString5(value)) {
        return "String";
      }
      if (isNumber4(value)) {
        return "Number";
      }
      if (isBoolean4(value)) {
        return "Boolean";
      }
      if (isSymbol3(value)) {
        return "Symbol";
      }
      if (isBigInt(value)) {
        return "BigInt";
      }
    };
  }
});

// node_modules/is-weakmap/index.js
var require_is_weakmap = __commonJS({
  "node_modules/is-weakmap/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var $WeakMap = typeof WeakMap === "function" && WeakMap.prototype ? WeakMap : null;
    var $WeakSet = typeof WeakSet === "function" && WeakSet.prototype ? WeakSet : null;
    var exported;
    if (!$WeakMap) {
      exported = function isWeakMap(x4) {
        return false;
      };
    }
    var $mapHas = $WeakMap ? $WeakMap.prototype.has : null;
    var $setHas = $WeakSet ? $WeakSet.prototype.has : null;
    if (!exported && !$mapHas) {
      exported = function isWeakMap(x4) {
        return false;
      };
    }
    module.exports = exported || function isWeakMap(x4) {
      if (!x4 || typeof x4 !== "object") {
        return false;
      }
      try {
        $mapHas.call(x4, $mapHas);
        if ($setHas) {
          try {
            $setHas.call(x4, $setHas);
          } catch (e10) {
            return true;
          }
        }
        return x4 instanceof $WeakMap;
      } catch (e10) {
      }
      return false;
    };
  }
});

// node_modules/is-weakset/index.js
var require_is_weakset = __commonJS({
  "node_modules/is-weakset/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var GetIntrinsic = require_get_intrinsic();
    var callBound = require_callBound();
    var $WeakSet = GetIntrinsic("%WeakSet%", true);
    var $setHas = callBound("WeakSet.prototype.has", true);
    if ($setHas) {
      $mapHas = callBound("WeakMap.prototype.has", true);
      module.exports = function isWeakSet(x4) {
        if (!x4 || typeof x4 !== "object") {
          return false;
        }
        try {
          $setHas(x4, $setHas);
          if ($mapHas) {
            try {
              $mapHas(x4, $mapHas);
            } catch (e10) {
              return true;
            }
          }
          return x4 instanceof $WeakSet;
        } catch (e10) {
        }
        return false;
      };
    } else {
      module.exports = function isWeakSet(x4) {
        return false;
      };
    }
    var $mapHas;
  }
});

// node_modules/which-collection/index.js
var require_which_collection = __commonJS({
  "node_modules/which-collection/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var isMap = require_is_map();
    var isSet = require_is_set();
    var isWeakMap = require_is_weakmap();
    var isWeakSet = require_is_weakset();
    module.exports = function whichCollection(value) {
      if (value && typeof value === "object") {
        if (isMap(value)) {
          return "Map";
        }
        if (isSet(value)) {
          return "Set";
        }
        if (isWeakMap(value)) {
          return "WeakMap";
        }
        if (isWeakSet(value)) {
          return "WeakSet";
        }
      }
      return false;
    };
  }
});

// node_modules/which-typed-array/index.js
var require_which_typed_array = __commonJS({
  "node_modules/which-typed-array/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var forEach2 = require_for_each();
    var availableTypedArrays = require_available_typed_arrays();
    var callBound = require_callBound();
    var gOPD = require_gopd();
    var $toString = callBound("Object.prototype.toString");
    var hasToStringTag = require_shams2()();
    var g5 = typeof globalThis === "undefined" ? global : globalThis;
    var typedArrays = availableTypedArrays();
    var $slice = callBound("String.prototype.slice");
    var toStrTags = {};
    var getPrototypeOf2 = Object.getPrototypeOf;
    if (hasToStringTag && gOPD && getPrototypeOf2) {
      forEach2(typedArrays, function(typedArray) {
        if (typeof g5[typedArray] === "function") {
          var arr = new g5[typedArray]();
          if (Symbol.toStringTag in arr) {
            var proto = getPrototypeOf2(arr);
            var descriptor = gOPD(proto, Symbol.toStringTag);
            if (!descriptor) {
              var superProto = getPrototypeOf2(proto);
              descriptor = gOPD(superProto, Symbol.toStringTag);
            }
            toStrTags[typedArray] = descriptor.get;
          }
        }
      });
    }
    var tryTypedArrays = function tryAllTypedArrays(value) {
      var foundName = false;
      forEach2(toStrTags, function(getter, typedArray) {
        if (!foundName) {
          try {
            var name3 = getter.call(value);
            if (name3 === typedArray) {
              foundName = name3;
            }
          } catch (e10) {
          }
        }
      });
      return foundName;
    };
    var isTypedArray2 = require_is_typed_array();
    module.exports = function whichTypedArray(value) {
      if (!isTypedArray2(value)) {
        return false;
      }
      if (!hasToStringTag || !(Symbol.toStringTag in value)) {
        return $slice($toString(value), 8, -1);
      }
      return tryTypedArrays(value);
    };
  }
});

// node_modules/array-buffer-byte-length/index.js
var require_array_buffer_byte_length = __commonJS({
  "node_modules/array-buffer-byte-length/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var callBound = require_callBound();
    var $byteLength = callBound("ArrayBuffer.prototype.byteLength", true);
    var isArrayBuffer2 = require_is_array_buffer();
    module.exports = function byteLength(ab) {
      if (!isArrayBuffer2(ab)) {
        return NaN;
      }
      return $byteLength ? $byteLength(ab) : ab.byteLength;
    };
  }
});

// node_modules/deep-equal/index.js
var require_deep_equal = __commonJS({
  "node_modules/deep-equal/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var assign = require_object();
    var callBound = require_callBound();
    var flags = require_regexp_prototype();
    var GetIntrinsic = require_get_intrinsic();
    var getIterator = require_es_get_iterator();
    var getSideChannel = require_side_channel();
    var is = require_object_is();
    var isArguments = require_is_arguments();
    var isArray4 = require_isarray();
    var isArrayBuffer2 = require_is_array_buffer();
    var isDate4 = require_is_date_object();
    var isRegex = require_is_regex();
    var isSharedArrayBuffer = require_is_shared_array_buffer();
    var objectKeys = require_object_keys();
    var whichBoxedPrimitive = require_which_boxed_primitive();
    var whichCollection = require_which_collection();
    var whichTypedArray = require_which_typed_array();
    var byteLength = require_array_buffer_byte_length();
    var sabByteLength = callBound("SharedArrayBuffer.prototype.byteLength", true);
    var $getTime = callBound("Date.prototype.getTime");
    var gPO = Object.getPrototypeOf;
    var $objToString = callBound("Object.prototype.toString");
    var $Set = GetIntrinsic("%Set%", true);
    var $mapHas = callBound("Map.prototype.has", true);
    var $mapGet = callBound("Map.prototype.get", true);
    var $mapSize = callBound("Map.prototype.size", true);
    var $setAdd = callBound("Set.prototype.add", true);
    var $setDelete = callBound("Set.prototype.delete", true);
    var $setHas = callBound("Set.prototype.has", true);
    var $setSize = callBound("Set.prototype.size", true);
    function setHasEqualElement(set, val1, opts, channel) {
      var i7 = getIterator(set);
      var result;
      while ((result = i7.next()) && !result.done) {
        if (internalDeepEqual(val1, result.value, opts, channel)) {
          $setDelete(set, result.value);
          return true;
        }
      }
      return false;
    }
    function findLooseMatchingPrimitives(prim) {
      if (typeof prim === "undefined") {
        return null;
      }
      if (typeof prim === "object") {
        return void 0;
      }
      if (typeof prim === "symbol") {
        return false;
      }
      if (typeof prim === "string" || typeof prim === "number") {
        return +prim === +prim;
      }
      return true;
    }
    function mapMightHaveLoosePrim(a7, b5, prim, item, opts, channel) {
      var altValue = findLooseMatchingPrimitives(prim);
      if (altValue != null) {
        return altValue;
      }
      var curB = $mapGet(b5, altValue);
      var looseOpts = assign({}, opts, { strict: false });
      if (typeof curB === "undefined" && !$mapHas(b5, altValue) || !internalDeepEqual(item, curB, looseOpts, channel)) {
        return false;
      }
      return !$mapHas(a7, altValue) && internalDeepEqual(item, curB, looseOpts, channel);
    }
    function setMightHaveLoosePrim(a7, b5, prim) {
      var altValue = findLooseMatchingPrimitives(prim);
      if (altValue != null) {
        return altValue;
      }
      return $setHas(b5, altValue) && !$setHas(a7, altValue);
    }
    function mapHasEqualEntry(set, map, key1, item1, opts, channel) {
      var i7 = getIterator(set);
      var result;
      var key2;
      while ((result = i7.next()) && !result.done) {
        key2 = result.value;
        if (
          // eslint-disable-next-line no-use-before-define
          internalDeepEqual(key1, key2, opts, channel) && internalDeepEqual(item1, $mapGet(map, key2), opts, channel)
        ) {
          $setDelete(set, key2);
          return true;
        }
      }
      return false;
    }
    function internalDeepEqual(actual, expected, options, channel) {
      var opts = options || {};
      if (opts.strict ? is(actual, expected) : actual === expected) {
        return true;
      }
      var actualBoxed = whichBoxedPrimitive(actual);
      var expectedBoxed = whichBoxedPrimitive(expected);
      if (actualBoxed !== expectedBoxed) {
        return false;
      }
      if (!actual || !expected || typeof actual !== "object" && typeof expected !== "object") {
        return opts.strict ? is(actual, expected) : actual == expected;
      }
      var hasActual = channel.has(actual);
      var hasExpected = channel.has(expected);
      var sentinel;
      if (hasActual && hasExpected) {
        if (channel.get(actual) === channel.get(expected)) {
          return true;
        }
      } else {
        sentinel = {};
      }
      if (!hasActual) {
        channel.set(actual, sentinel);
      }
      if (!hasExpected) {
        channel.set(expected, sentinel);
      }
      return objEquiv(actual, expected, opts, channel);
    }
    function isBuffer4(x4) {
      if (!x4 || typeof x4 !== "object" || typeof x4.length !== "number") {
        return false;
      }
      if (typeof x4.copy !== "function" || typeof x4.slice !== "function") {
        return false;
      }
      if (x4.length > 0 && typeof x4[0] !== "number") {
        return false;
      }
      return !!(x4.constructor && x4.constructor.isBuffer && x4.constructor.isBuffer(x4));
    }
    function setEquiv(a7, b5, opts, channel) {
      if ($setSize(a7) !== $setSize(b5)) {
        return false;
      }
      var iA = getIterator(a7);
      var iB = getIterator(b5);
      var resultA;
      var resultB;
      var set;
      while ((resultA = iA.next()) && !resultA.done) {
        if (resultA.value && typeof resultA.value === "object") {
          if (!set) {
            set = new $Set();
          }
          $setAdd(set, resultA.value);
        } else if (!$setHas(b5, resultA.value)) {
          if (opts.strict) {
            return false;
          }
          if (!setMightHaveLoosePrim(a7, b5, resultA.value)) {
            return false;
          }
          if (!set) {
            set = new $Set();
          }
          $setAdd(set, resultA.value);
        }
      }
      if (set) {
        while ((resultB = iB.next()) && !resultB.done) {
          if (resultB.value && typeof resultB.value === "object") {
            if (!setHasEqualElement(set, resultB.value, opts.strict, channel)) {
              return false;
            }
          } else if (!opts.strict && !$setHas(a7, resultB.value) && !setHasEqualElement(set, resultB.value, opts.strict, channel)) {
            return false;
          }
        }
        return $setSize(set) === 0;
      }
      return true;
    }
    function mapEquiv(a7, b5, opts, channel) {
      if ($mapSize(a7) !== $mapSize(b5)) {
        return false;
      }
      var iA = getIterator(a7);
      var iB = getIterator(b5);
      var resultA;
      var resultB;
      var set;
      var key;
      var item1;
      var item2;
      while ((resultA = iA.next()) && !resultA.done) {
        key = resultA.value[0];
        item1 = resultA.value[1];
        if (key && typeof key === "object") {
          if (!set) {
            set = new $Set();
          }
          $setAdd(set, key);
        } else {
          item2 = $mapGet(b5, key);
          if (typeof item2 === "undefined" && !$mapHas(b5, key) || !internalDeepEqual(item1, item2, opts, channel)) {
            if (opts.strict) {
              return false;
            }
            if (!mapMightHaveLoosePrim(a7, b5, key, item1, opts, channel)) {
              return false;
            }
            if (!set) {
              set = new $Set();
            }
            $setAdd(set, key);
          }
        }
      }
      if (set) {
        while ((resultB = iB.next()) && !resultB.done) {
          key = resultB.value[0];
          item2 = resultB.value[1];
          if (key && typeof key === "object") {
            if (!mapHasEqualEntry(set, a7, key, item2, opts, channel)) {
              return false;
            }
          } else if (!opts.strict && (!a7.has(key) || !internalDeepEqual($mapGet(a7, key), item2, opts, channel)) && !mapHasEqualEntry(set, a7, key, item2, assign({}, opts, { strict: false }), channel)) {
            return false;
          }
        }
        return $setSize(set) === 0;
      }
      return true;
    }
    function objEquiv(a7, b5, opts, channel) {
      var i7, key;
      if (typeof a7 !== typeof b5) {
        return false;
      }
      if (a7 == null || b5 == null) {
        return false;
      }
      if ($objToString(a7) !== $objToString(b5)) {
        return false;
      }
      if (isArguments(a7) !== isArguments(b5)) {
        return false;
      }
      var aIsArray = isArray4(a7);
      var bIsArray = isArray4(b5);
      if (aIsArray !== bIsArray) {
        return false;
      }
      var aIsError = a7 instanceof Error;
      var bIsError = b5 instanceof Error;
      if (aIsError !== bIsError) {
        return false;
      }
      if (aIsError || bIsError) {
        if (a7.name !== b5.name || a7.message !== b5.message) {
          return false;
        }
      }
      var aIsRegex = isRegex(a7);
      var bIsRegex = isRegex(b5);
      if (aIsRegex !== bIsRegex) {
        return false;
      }
      if ((aIsRegex || bIsRegex) && (a7.source !== b5.source || flags(a7) !== flags(b5))) {
        return false;
      }
      var aIsDate = isDate4(a7);
      var bIsDate = isDate4(b5);
      if (aIsDate !== bIsDate) {
        return false;
      }
      if (aIsDate || bIsDate) {
        if ($getTime(a7) !== $getTime(b5)) {
          return false;
        }
      }
      if (opts.strict && gPO && gPO(a7) !== gPO(b5)) {
        return false;
      }
      var aWhich = whichTypedArray(a7);
      var bWhich = whichTypedArray(b5);
      if ((aWhich || bWhich) && aWhich !== bWhich) {
        return false;
      }
      var aIsBuffer = isBuffer4(a7);
      var bIsBuffer = isBuffer4(b5);
      if (aIsBuffer !== bIsBuffer) {
        return false;
      }
      if (aIsBuffer || bIsBuffer) {
        if (a7.length !== b5.length) {
          return false;
        }
        for (i7 = 0; i7 < a7.length; i7++) {
          if (a7[i7] !== b5[i7]) {
            return false;
          }
        }
        return true;
      }
      var aIsArrayBuffer = isArrayBuffer2(a7);
      var bIsArrayBuffer = isArrayBuffer2(b5);
      if (aIsArrayBuffer !== bIsArrayBuffer) {
        return false;
      }
      if (aIsArrayBuffer || bIsArrayBuffer) {
        if (byteLength(a7) !== byteLength(b5)) {
          return false;
        }
        return typeof Uint8Array === "function" && internalDeepEqual(new Uint8Array(a7), new Uint8Array(b5), opts, channel);
      }
      var aIsSAB = isSharedArrayBuffer(a7);
      var bIsSAB = isSharedArrayBuffer(b5);
      if (aIsSAB !== bIsSAB) {
        return false;
      }
      if (aIsSAB || bIsSAB) {
        if (sabByteLength(a7) !== sabByteLength(b5)) {
          return false;
        }
        return typeof Uint8Array === "function" && internalDeepEqual(new Uint8Array(a7), new Uint8Array(b5), opts, channel);
      }
      if (typeof a7 !== typeof b5) {
        return false;
      }
      var ka = objectKeys(a7);
      var kb = objectKeys(b5);
      if (ka.length !== kb.length) {
        return false;
      }
      ka.sort();
      kb.sort();
      for (i7 = ka.length - 1; i7 >= 0; i7--) {
        if (ka[i7] != kb[i7]) {
          return false;
        }
      }
      for (i7 = ka.length - 1; i7 >= 0; i7--) {
        key = ka[i7];
        if (!internalDeepEqual(a7[key], b5[key], opts, channel)) {
          return false;
        }
      }
      var aCollection = whichCollection(a7);
      var bCollection = whichCollection(b5);
      if (aCollection !== bCollection) {
        return false;
      }
      if (aCollection === "Set" || bCollection === "Set") {
        return setEquiv(a7, b5, opts, channel);
      }
      if (aCollection === "Map") {
        return mapEquiv(a7, b5, opts, channel);
      }
      return true;
    }
    module.exports = function deepEqual3(a7, b5, opts) {
      return internalDeepEqual(a7, b5, opts, getSideChannel());
    };
  }
});

// node_modules/deepmerge/dist/cjs.js
var require_cjs = __commonJS({
  "node_modules/deepmerge/dist/cjs.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var isMergeableObject = function isMergeableObject2(value) {
      return isNonNullObject(value) && !isSpecial(value);
    };
    function isNonNullObject(value) {
      return !!value && typeof value === "object";
    }
    function isSpecial(value) {
      var stringValue = Object.prototype.toString.call(value);
      return stringValue === "[object RegExp]" || stringValue === "[object Date]" || isReactElement(value);
    }
    var canUseSymbol = typeof Symbol === "function" && Symbol.for;
    var REACT_ELEMENT_TYPE = canUseSymbol ? Symbol.for("react.element") : 60103;
    function isReactElement(value) {
      return value.$$typeof === REACT_ELEMENT_TYPE;
    }
    function emptyTarget(val) {
      return Array.isArray(val) ? [] : {};
    }
    function cloneUnlessOtherwiseSpecified(value, options) {
      return options.clone !== false && options.isMergeableObject(value) ? deepmerge(emptyTarget(value), value, options) : value;
    }
    function defaultArrayMerge(target, source, options) {
      return target.concat(source).map(function(element) {
        return cloneUnlessOtherwiseSpecified(element, options);
      });
    }
    function getMergeFunction(key, options) {
      if (!options.customMerge) {
        return deepmerge;
      }
      var customMerge = options.customMerge(key);
      return typeof customMerge === "function" ? customMerge : deepmerge;
    }
    function getEnumerableOwnPropertySymbols(target) {
      return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(target).filter(function(symbol) {
        return Object.propertyIsEnumerable.call(target, symbol);
      }) : [];
    }
    function getKeys(target) {
      return Object.keys(target).concat(getEnumerableOwnPropertySymbols(target));
    }
    function propertyIsOnObject(object, property) {
      try {
        return property in object;
      } catch (_4) {
        return false;
      }
    }
    function propertyIsUnsafe(target, key) {
      return propertyIsOnObject(target, key) && !(Object.hasOwnProperty.call(target, key) && Object.propertyIsEnumerable.call(target, key));
    }
    function mergeObject(target, source, options) {
      var destination = {};
      if (options.isMergeableObject(target)) {
        getKeys(target).forEach(function(key) {
          destination[key] = cloneUnlessOtherwiseSpecified(target[key], options);
        });
      }
      getKeys(source).forEach(function(key) {
        if (propertyIsUnsafe(target, key)) {
          return;
        }
        if (propertyIsOnObject(target, key) && options.isMergeableObject(source[key])) {
          destination[key] = getMergeFunction(key, options)(target[key], source[key], options);
        } else {
          destination[key] = cloneUnlessOtherwiseSpecified(source[key], options);
        }
      });
      return destination;
    }
    function deepmerge(target, source, options) {
      options = options || {};
      options.arrayMerge = options.arrayMerge || defaultArrayMerge;
      options.isMergeableObject = options.isMergeableObject || isMergeableObject;
      options.cloneUnlessOtherwiseSpecified = cloneUnlessOtherwiseSpecified;
      var sourceIsArray = Array.isArray(source);
      var targetIsArray = Array.isArray(target);
      var sourceAndTargetTypesMatch = sourceIsArray === targetIsArray;
      if (!sourceAndTargetTypesMatch) {
        return cloneUnlessOtherwiseSpecified(source, options);
      } else if (sourceIsArray) {
        return options.arrayMerge(target, source, options);
      } else {
        return mergeObject(target, source, options);
      }
    }
    deepmerge.all = function deepmergeAll(array, options) {
      if (!Array.isArray(array)) {
        throw new Error("first argument should be an array");
      }
      return array.reduce(function(prev, next) {
        return deepmerge(prev, next, options);
      }, {});
    };
    var deepmerge_1 = deepmerge;
    module.exports = deepmerge_1;
  }
});

// node_modules/form-data/lib/browser.js
var require_browser = __commonJS({
  "node_modules/form-data/lib/browser.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    module.exports = typeof self == "object" ? self.FormData : window.FormData;
  }
});
var exports3;

// node_modules/@jspm/core/nodelibs/browser/path.js
var path_exports = {};
__export(path_exports, {
  _makeLong: () => _makeLong,
  basename: () => basename,
  default: () => exports3,
  delimiter: () => delimiter,
  dirname: () => dirname,
  extname: () => extname,
  format: () => format,
  isAbsolute: () => isAbsolute,
  join: () => join,
  normalize: () => normalize,
  parse: () => parse,
  posix: () => posix,
  relative: () => relative,
  resolve: () => resolve2,
  sep: () => sep,
  win32: () => win32
});
var _makeLong, basename, delimiter, dirname, extname, format, isAbsolute, join, normalize, parse, posix, relative, resolve2, sep, win32;

// node_modules/@jspm/core/nodelibs/browser/os.js
var os_exports = {};
__export(os_exports, {
  EOL: () => EOL,
  arch: () => arch2,
  constants: () => constants,
  cpus: () => cpus,
  default: () => exports4,
  endianness: () => endianness,
  freemem: () => freemem,
  getNetworkInterfaces: () => getNetworkInterfaces,
  homedir: () => homedir,
  hostname: () => hostname,
  loadavg: () => loadavg,
  networkInterfaces: () => networkInterfaces,
  platform: () => platform2,
  release: () => release2,
  tmpDir: () => tmpDir,
  tmpdir: () => tmpdir,
  totalmem: () => totalmem,
  type: () => type,
  uptime: () => uptime,
  version: () => version2
});
var exports4, version2, constants, EOL, arch2, cpus, endianness, freemem, getNetworkInterfaces, homedir, hostname, loadavg, networkInterfaces, platform2, release2, tmpDir, tmpdir, totalmem, type;
var X;

// node_modules/@jspm/core/nodelibs/browser/assert.js
var assert_exports = {};
__export(assert_exports, {
  AssertionError: () => AssertionError,
  deepEqual: () => deepEqual,
  deepStrictEqual: () => deepStrictEqual,
  default: () => et,
  doesNotReject: () => doesNotReject,
  doesNotThrow: () => doesNotThrow,
  equal: () => equal,
  fail: () => fail,
  ifError: () => ifError,
  notDeepEqual: () => notDeepEqual,
  notDeepStrictEqual: () => notDeepStrictEqual,
  notEqual: () => notEqual,
  notStrictEqual: () => notStrictEqual,
  ok: () => ok,
  rejects: () => rejects,
  strict: () => strict,
  strictEqual: () => strictEqual,
  throws: () => throws
});
var et, AssertionError, deepEqual, deepStrictEqual, doesNotReject, doesNotThrow, equal, fail, ifError, notDeepEqual, notDeepStrictEqual, notEqual, notStrictEqual, ok, rejects, strict, strictEqual, throws;

// node_modules/@jspm/core/nodelibs/browser/util.js
var util_exports = {};
__export(util_exports, {
  TextDecoder: () => TextDecoder3,
  TextEncoder: () => TextEncoder2,
  _extend: () => _extend2,
  callbackify: () => callbackify2,
  debuglog: () => debuglog2,
  default: () => X,
  deprecate: () => deprecate2,
  format: () => format3,
  inherits: () => inherits3,
  inspect: () => inspect2,
  isArray: () => isArray3,
  isBoolean: () => isBoolean3,
  isBuffer: () => isBuffer3,
  isDate: () => isDate3,
  isError: () => isError2,
  isFunction: () => isFunction3,
  isNull: () => isNull2,
  isNullOrUndefined: () => isNullOrUndefined2,
  isNumber: () => isNumber3,
  isObject: () => isObject3,
  isPrimitive: () => isPrimitive2,
  isRegExp: () => isRegExp3,
  isString: () => isString4,
  isSymbol: () => isSymbol2,
  isUndefined: () => isUndefined3,
  log: () => log2,
  promisify: () => promisify2,
  types: () => types2
});
var _extend2, callbackify2, debuglog2, deprecate2, format3, inherits3, inspect2, isArray3, isBoolean3, isBuffer3, isDate3, isError2, isFunction3, isNull2, isNullOrUndefined2, isNumber3, isObject3, isPrimitive2, isRegExp3, isString4, isSymbol2, isUndefined3, log2, promisify2, types2, TextEncoder2, TextDecoder3;
var exports6;

// node_modules/@jspm/core/nodelibs/browser/stream.js
var stream_exports = {};
__export(stream_exports, {
  Duplex: () => Duplex,
  PassThrough: () => PassThrough,
  Readable: () => Readable,
  Stream: () => Stream,
  Transform: () => Transform,
  Writable: () => Writable,
  default: () => exports6,
  finished: () => finished,
  pipeline: () => pipeline,
  promises: () => promises
});
var Readable, Writable, Duplex, Transform, PassThrough, finished, pipeline, Stream, promises;

// node_modules/@jspm/core/nodelibs/browser/fs.js
var fs_exports = {};
__export(fs_exports, {
  Dir: () => Dir,
  Dirent: () => Dirent,
  F_OK: () => F_OK,
  FileReadStream: () => FileReadStream,
  FileWriteStream: () => FileWriteStream,
  R_OK: () => R_OK,
  ReadStream: () => ReadStream,
  Stats: () => Stats,
  W_OK: () => W_OK,
  WriteStream: () => WriteStream,
  X_OK: () => X_OK,
  _toUnixTimestamp: () => _toUnixTimestamp,
  access: () => access,
  accessSync: () => accessSync,
  appendFile: () => appendFile,
  appendFileSync: () => appendFileSync,
  chmod: () => chmod,
  chmodSync: () => chmodSync,
  chown: () => chown,
  chownSync: () => chownSync,
  close: () => close,
  closeSync: () => closeSync,
  constants: () => constants2,
  copyFile: () => copyFile,
  copyFileSync: () => copyFileSync,
  cp: () => cp,
  cpSync: () => cpSync,
  createReadStream: () => createReadStream,
  createWriteStream: () => createWriteStream,
  default: () => fs,
  exists: () => exists,
  existsSync: () => existsSync,
  fchmod: () => fchmod,
  fchmodSync: () => fchmodSync,
  fchown: () => fchown,
  fchownSync: () => fchownSync,
  fdatasync: () => fdatasync,
  fdatasyncSync: () => fdatasyncSync,
  fstat: () => fstat,
  fstatSync: () => fstatSync,
  fsync: () => fsync,
  fsyncSync: () => fsyncSync,
  ftruncate: () => ftruncate,
  ftruncateSync: () => ftruncateSync,
  futimes: () => futimes,
  futimesSync: () => futimesSync,
  lchmod: () => lchmod,
  lchmodSync: () => lchmodSync,
  lchown: () => lchown,
  lchownSync: () => lchownSync,
  link: () => link,
  linkSync: () => linkSync,
  lstat: () => lstat,
  lstatSync: () => lstatSync,
  mkdir: () => mkdir,
  mkdirSync: () => mkdirSync,
  mkdtemp: () => mkdtemp,
  mkdtempSync: () => mkdtempSync,
  open: () => open,
  openSync: () => openSync,
  opendir: () => opendir,
  opendirSync: () => opendirSync,
  promises: () => promises2,
  read: () => read,
  readFile: () => readFile,
  readFileSync: () => readFileSync,
  readSync: () => readSync,
  readdir: () => readdir,
  readdirSync: () => readdirSync,
  readlink: () => readlink,
  readlinkSync: () => readlinkSync,
  readv: () => readv,
  readvSync: () => readvSync,
  realpath: () => realpath,
  realpathSync: () => realpathSync,
  rename: () => rename,
  renameSync: () => renameSync,
  rm: () => rm,
  rmSync: () => rmSync,
  rmdir: () => rmdir,
  rmdirSync: () => rmdirSync,
  stat: () => stat,
  statSync: () => statSync,
  symlink: () => symlink,
  symlinkSync: () => symlinkSync,
  truncate: () => truncate,
  truncateSync: () => truncateSync,
  unlink: () => unlink,
  unlinkSync: () => unlinkSync,
  unwatchFile: () => unwatchFile,
  utimes: () => utimes,
  utimesSync: () => utimesSync,
  watch: () => watch,
  watchFile: () => watchFile,
  write: () => write,
  writeFile: () => writeFile,
  writeFileSync: () => writeFileSync,
  writeSync: () => writeSync,
  writev: () => writev,
  writevSync: () => writevSync
});
var fs, appendFile, appendFileSync, access, accessSync, chown, chownSync, chmod, chmodSync, close, closeSync, copyFile, copyFileSync, cp, cpSync, createReadStream, createWriteStream, exists, existsSync, fchown, fchownSync, fchmod, fchmodSync, fdatasync, fdatasyncSync, fstat, fstatSync, fsync, fsyncSync, ftruncate, ftruncateSync, futimes, futimesSync, lchown, lchownSync, lchmod, lchmodSync, link, linkSync, lstat, lstatSync, mkdir, mkdirSync, mkdtemp, mkdtempSync, open, openSync, opendir, opendirSync, readdir, readdirSync, read, readSync, readv, readvSync, readFile, readFileSync, readlink, readlinkSync, realpath, realpathSync, rename, renameSync, rm, rmSync, rmdir, rmdirSync, stat, statSync, symlink, symlinkSync, truncate, truncateSync, unwatchFile, unlink, unlinkSync, utimes, utimesSync, watch, watchFile, writeFile, writeFileSync, write, writeSync, writev, writevSync, Dir, Dirent, Stats, ReadStream, WriteStream, FileReadStream, FileWriteStream, _toUnixTimestamp, F_OK, R_OK, W_OK, X_OK, constants2, promises2;

// node_modules/@jspm/core/nodelibs/browser/constants.js
var constants_exports = {};
__export(constants_exports, {
  DH_CHECK_P_NOT_PRIME: () => DH_CHECK_P_NOT_PRIME,
  DH_CHECK_P_NOT_SAFE_PRIME: () => DH_CHECK_P_NOT_SAFE_PRIME,
  DH_NOT_SUITABLE_GENERATOR: () => DH_NOT_SUITABLE_GENERATOR,
  DH_UNABLE_TO_CHECK_GENERATOR: () => DH_UNABLE_TO_CHECK_GENERATOR,
  E2BIG: () => E2BIG,
  EACCES: () => EACCES,
  EADDRINUSE: () => EADDRINUSE,
  EADDRNOTAVAIL: () => EADDRNOTAVAIL,
  EAFNOSUPPORT: () => EAFNOSUPPORT,
  EAGAIN: () => EAGAIN,
  EALREADY: () => EALREADY,
  EBADF: () => EBADF,
  EBADMSG: () => EBADMSG,
  EBUSY: () => EBUSY,
  ECANCELED: () => ECANCELED,
  ECHILD: () => ECHILD,
  ECONNABORTED: () => ECONNABORTED,
  ECONNREFUSED: () => ECONNREFUSED,
  ECONNRESET: () => ECONNRESET,
  EDEADLK: () => EDEADLK,
  EDESTADDRREQ: () => EDESTADDRREQ,
  EDOM: () => EDOM,
  EDQUOT: () => EDQUOT,
  EEXIST: () => EEXIST,
  EFAULT: () => EFAULT,
  EFBIG: () => EFBIG,
  EHOSTUNREACH: () => EHOSTUNREACH,
  EIDRM: () => EIDRM,
  EILSEQ: () => EILSEQ,
  EINPROGRESS: () => EINPROGRESS,
  EINTR: () => EINTR,
  EINVAL: () => EINVAL,
  EIO: () => EIO,
  EISCONN: () => EISCONN,
  EISDIR: () => EISDIR,
  ELOOP: () => ELOOP,
  EMFILE: () => EMFILE,
  EMLINK: () => EMLINK,
  EMSGSIZE: () => EMSGSIZE,
  EMULTIHOP: () => EMULTIHOP,
  ENAMETOOLONG: () => ENAMETOOLONG,
  ENETDOWN: () => ENETDOWN,
  ENETRESET: () => ENETRESET,
  ENETUNREACH: () => ENETUNREACH,
  ENFILE: () => ENFILE,
  ENGINE_METHOD_ALL: () => ENGINE_METHOD_ALL,
  ENGINE_METHOD_CIPHERS: () => ENGINE_METHOD_CIPHERS,
  ENGINE_METHOD_DH: () => ENGINE_METHOD_DH,
  ENGINE_METHOD_DIGESTS: () => ENGINE_METHOD_DIGESTS,
  ENGINE_METHOD_DSA: () => ENGINE_METHOD_DSA,
  ENGINE_METHOD_ECDH: () => ENGINE_METHOD_ECDH,
  ENGINE_METHOD_ECDSA: () => ENGINE_METHOD_ECDSA,
  ENGINE_METHOD_NONE: () => ENGINE_METHOD_NONE,
  ENGINE_METHOD_PKEY_ASN1_METHS: () => ENGINE_METHOD_PKEY_ASN1_METHS,
  ENGINE_METHOD_PKEY_METHS: () => ENGINE_METHOD_PKEY_METHS,
  ENGINE_METHOD_RAND: () => ENGINE_METHOD_RAND,
  ENGINE_METHOD_STORE: () => ENGINE_METHOD_STORE,
  ENOBUFS: () => ENOBUFS,
  ENODATA: () => ENODATA,
  ENODEV: () => ENODEV,
  ENOENT: () => ENOENT,
  ENOEXEC: () => ENOEXEC,
  ENOLCK: () => ENOLCK,
  ENOLINK: () => ENOLINK,
  ENOMEM: () => ENOMEM,
  ENOMSG: () => ENOMSG,
  ENOPROTOOPT: () => ENOPROTOOPT,
  ENOSPC: () => ENOSPC,
  ENOSR: () => ENOSR,
  ENOSTR: () => ENOSTR,
  ENOSYS: () => ENOSYS,
  ENOTCONN: () => ENOTCONN,
  ENOTDIR: () => ENOTDIR,
  ENOTEMPTY: () => ENOTEMPTY,
  ENOTSOCK: () => ENOTSOCK,
  ENOTSUP: () => ENOTSUP,
  ENOTTY: () => ENOTTY,
  ENXIO: () => ENXIO,
  EOPNOTSUPP: () => EOPNOTSUPP,
  EOVERFLOW: () => EOVERFLOW,
  EPERM: () => EPERM,
  EPIPE: () => EPIPE,
  EPROTO: () => EPROTO,
  EPROTONOSUPPORT: () => EPROTONOSUPPORT,
  EPROTOTYPE: () => EPROTOTYPE,
  ERANGE: () => ERANGE,
  EROFS: () => EROFS,
  ESPIPE: () => ESPIPE,
  ESRCH: () => ESRCH,
  ESTALE: () => ESTALE,
  ETIME: () => ETIME,
  ETIMEDOUT: () => ETIMEDOUT,
  ETXTBSY: () => ETXTBSY,
  EWOULDBLOCK: () => EWOULDBLOCK,
  EXDEV: () => EXDEV,
  F_OK: () => F_OK2,
  NPN_ENABLED: () => NPN_ENABLED,
  O_APPEND: () => O_APPEND,
  O_CREAT: () => O_CREAT,
  O_DIRECTORY: () => O_DIRECTORY,
  O_EXCL: () => O_EXCL,
  O_NOCTTY: () => O_NOCTTY,
  O_NOFOLLOW: () => O_NOFOLLOW,
  O_NONBLOCK: () => O_NONBLOCK,
  O_RDONLY: () => O_RDONLY,
  O_RDWR: () => O_RDWR,
  O_SYMLINK: () => O_SYMLINK,
  O_SYNC: () => O_SYNC,
  O_TRUNC: () => O_TRUNC,
  O_WRONLY: () => O_WRONLY,
  POINT_CONVERSION_COMPRESSED: () => POINT_CONVERSION_COMPRESSED,
  POINT_CONVERSION_HYBRID: () => POINT_CONVERSION_HYBRID,
  POINT_CONVERSION_UNCOMPRESSED: () => POINT_CONVERSION_UNCOMPRESSED,
  RSA_NO_PADDING: () => RSA_NO_PADDING,
  RSA_PKCS1_OAEP_PADDING: () => RSA_PKCS1_OAEP_PADDING,
  RSA_PKCS1_PADDING: () => RSA_PKCS1_PADDING,
  RSA_PKCS1_PSS_PADDING: () => RSA_PKCS1_PSS_PADDING,
  RSA_SSLV23_PADDING: () => RSA_SSLV23_PADDING,
  RSA_X931_PADDING: () => RSA_X931_PADDING,
  R_OK: () => R_OK2,
  SIGABRT: () => SIGABRT,
  SIGALRM: () => SIGALRM,
  SIGBUS: () => SIGBUS,
  SIGCHLD: () => SIGCHLD,
  SIGCONT: () => SIGCONT,
  SIGFPE: () => SIGFPE,
  SIGHUP: () => SIGHUP,
  SIGILL: () => SIGILL,
  SIGINT: () => SIGINT,
  SIGIO: () => SIGIO,
  SIGIOT: () => SIGIOT,
  SIGKILL: () => SIGKILL,
  SIGPIPE: () => SIGPIPE,
  SIGPROF: () => SIGPROF,
  SIGQUIT: () => SIGQUIT,
  SIGSEGV: () => SIGSEGV,
  SIGSTOP: () => SIGSTOP,
  SIGSYS: () => SIGSYS,
  SIGTERM: () => SIGTERM,
  SIGTRAP: () => SIGTRAP,
  SIGTSTP: () => SIGTSTP,
  SIGTTIN: () => SIGTTIN,
  SIGTTOU: () => SIGTTOU,
  SIGURG: () => SIGURG,
  SIGUSR1: () => SIGUSR1,
  SIGUSR2: () => SIGUSR2,
  SIGVTALRM: () => SIGVTALRM,
  SIGWINCH: () => SIGWINCH,
  SIGXCPU: () => SIGXCPU,
  SIGXFSZ: () => SIGXFSZ,
  SSL_OP_ALL: () => SSL_OP_ALL,
  SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION: () => SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION,
  SSL_OP_CIPHER_SERVER_PREFERENCE: () => SSL_OP_CIPHER_SERVER_PREFERENCE,
  SSL_OP_CISCO_ANYCONNECT: () => SSL_OP_CISCO_ANYCONNECT,
  SSL_OP_COOKIE_EXCHANGE: () => SSL_OP_COOKIE_EXCHANGE,
  SSL_OP_CRYPTOPRO_TLSEXT_BUG: () => SSL_OP_CRYPTOPRO_TLSEXT_BUG,
  SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS: () => SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS,
  SSL_OP_EPHEMERAL_RSA: () => SSL_OP_EPHEMERAL_RSA,
  SSL_OP_LEGACY_SERVER_CONNECT: () => SSL_OP_LEGACY_SERVER_CONNECT,
  SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER: () => SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER,
  SSL_OP_MICROSOFT_SESS_ID_BUG: () => SSL_OP_MICROSOFT_SESS_ID_BUG,
  SSL_OP_MSIE_SSLV2_RSA_PADDING: () => SSL_OP_MSIE_SSLV2_RSA_PADDING,
  SSL_OP_NETSCAPE_CA_DN_BUG: () => SSL_OP_NETSCAPE_CA_DN_BUG,
  SSL_OP_NETSCAPE_CHALLENGE_BUG: () => SSL_OP_NETSCAPE_CHALLENGE_BUG,
  SSL_OP_NETSCAPE_DEMO_CIPHER_CHANGE_BUG: () => SSL_OP_NETSCAPE_DEMO_CIPHER_CHANGE_BUG,
  SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG: () => SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG,
  SSL_OP_NO_COMPRESSION: () => SSL_OP_NO_COMPRESSION,
  SSL_OP_NO_QUERY_MTU: () => SSL_OP_NO_QUERY_MTU,
  SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION: () => SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION,
  SSL_OP_NO_SSLv2: () => SSL_OP_NO_SSLv2,
  SSL_OP_NO_SSLv3: () => SSL_OP_NO_SSLv3,
  SSL_OP_NO_TICKET: () => SSL_OP_NO_TICKET,
  SSL_OP_NO_TLSv1: () => SSL_OP_NO_TLSv1,
  SSL_OP_NO_TLSv1_1: () => SSL_OP_NO_TLSv1_1,
  SSL_OP_NO_TLSv1_2: () => SSL_OP_NO_TLSv1_2,
  SSL_OP_PKCS1_CHECK_1: () => SSL_OP_PKCS1_CHECK_1,
  SSL_OP_PKCS1_CHECK_2: () => SSL_OP_PKCS1_CHECK_2,
  SSL_OP_SINGLE_DH_USE: () => SSL_OP_SINGLE_DH_USE,
  SSL_OP_SINGLE_ECDH_USE: () => SSL_OP_SINGLE_ECDH_USE,
  SSL_OP_SSLEAY_080_CLIENT_DH_BUG: () => SSL_OP_SSLEAY_080_CLIENT_DH_BUG,
  SSL_OP_SSLREF2_REUSE_CERT_TYPE_BUG: () => SSL_OP_SSLREF2_REUSE_CERT_TYPE_BUG,
  SSL_OP_TLS_BLOCK_PADDING_BUG: () => SSL_OP_TLS_BLOCK_PADDING_BUG,
  SSL_OP_TLS_D5_BUG: () => SSL_OP_TLS_D5_BUG,
  SSL_OP_TLS_ROLLBACK_BUG: () => SSL_OP_TLS_ROLLBACK_BUG,
  S_IFBLK: () => S_IFBLK,
  S_IFCHR: () => S_IFCHR,
  S_IFDIR: () => S_IFDIR,
  S_IFIFO: () => S_IFIFO,
  S_IFLNK: () => S_IFLNK,
  S_IFMT: () => S_IFMT,
  S_IFREG: () => S_IFREG,
  S_IFSOCK: () => S_IFSOCK,
  S_IRGRP: () => S_IRGRP,
  S_IROTH: () => S_IROTH,
  S_IRUSR: () => S_IRUSR,
  S_IRWXG: () => S_IRWXG,
  S_IRWXO: () => S_IRWXO,
  S_IRWXU: () => S_IRWXU,
  S_IWGRP: () => S_IWGRP,
  S_IWOTH: () => S_IWOTH,
  S_IWUSR: () => S_IWUSR,
  S_IXGRP: () => S_IXGRP,
  S_IXOTH: () => S_IXOTH,
  S_IXUSR: () => S_IXUSR,
  UV_UDP_REUSEADDR: () => UV_UDP_REUSEADDR,
  W_OK: () => W_OK2,
  X_OK: () => X_OK2,
  default: () => constants3
});
var constants3, DH_CHECK_P_NOT_PRIME, DH_CHECK_P_NOT_SAFE_PRIME, DH_NOT_SUITABLE_GENERATOR, DH_UNABLE_TO_CHECK_GENERATOR, E2BIG, EACCES, EADDRINUSE, EADDRNOTAVAIL, EAFNOSUPPORT, EAGAIN, EALREADY, EBADF, EBADMSG, EBUSY, ECANCELED, ECHILD, ECONNABORTED, ECONNREFUSED, ECONNRESET, EDEADLK, EDESTADDRREQ, EDOM, EDQUOT, EEXIST, EFAULT, EFBIG, EHOSTUNREACH, EIDRM, EILSEQ, EINPROGRESS, EINTR, EINVAL, EIO, EISCONN, EISDIR, ELOOP, EMFILE, EMLINK, EMSGSIZE, EMULTIHOP, ENAMETOOLONG, ENETDOWN, ENETRESET, ENETUNREACH, ENFILE, ENGINE_METHOD_ALL, ENGINE_METHOD_CIPHERS, ENGINE_METHOD_DH, ENGINE_METHOD_DIGESTS, ENGINE_METHOD_DSA, ENGINE_METHOD_ECDH, ENGINE_METHOD_ECDSA, ENGINE_METHOD_NONE, ENGINE_METHOD_PKEY_ASN1_METHS, ENGINE_METHOD_PKEY_METHS, ENGINE_METHOD_RAND, ENGINE_METHOD_STORE, ENOBUFS, ENODATA, ENODEV, ENOENT, ENOEXEC, ENOLCK, ENOLINK, ENOMEM, ENOMSG, ENOPROTOOPT, ENOSPC, ENOSR, ENOSTR, ENOSYS, ENOTCONN, ENOTDIR, ENOTEMPTY, ENOTSOCK, ENOTSUP, ENOTTY, ENXIO, EOPNOTSUPP, EOVERFLOW, EPERM, EPIPE, EPROTO, EPROTONOSUPPORT, EPROTOTYPE, ERANGE, EROFS, ESPIPE, ESRCH, ESTALE, ETIME, ETIMEDOUT, ETXTBSY, EWOULDBLOCK, EXDEV, F_OK2, NPN_ENABLED, O_APPEND, O_CREAT, O_DIRECTORY, O_EXCL, O_NOCTTY, O_NOFOLLOW, O_NONBLOCK, O_RDONLY, O_RDWR, O_SYMLINK, O_SYNC, O_TRUNC, O_WRONLY, POINT_CONVERSION_COMPRESSED, POINT_CONVERSION_HYBRID, POINT_CONVERSION_UNCOMPRESSED, RSA_NO_PADDING, RSA_PKCS1_OAEP_PADDING, RSA_PKCS1_PADDING, RSA_PKCS1_PSS_PADDING, RSA_SSLV23_PADDING, RSA_X931_PADDING, R_OK2, SIGABRT, SIGALRM, SIGBUS, SIGCHLD, SIGCONT, SIGFPE, SIGHUP, SIGILL, SIGINT, SIGIO, SIGIOT, SIGKILL, SIGPIPE, SIGPROF, SIGQUIT, SIGSEGV, SIGSTOP, SIGSYS, SIGTERM, SIGTRAP, SIGTSTP, SIGTTIN, SIGTTOU, SIGURG, SIGUSR1, SIGUSR2, SIGVTALRM, SIGWINCH, SIGXCPU, SIGXFSZ, SSL_OP_ALL, SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION, SSL_OP_CIPHER_SERVER_PREFERENCE, SSL_OP_CISCO_ANYCONNECT, SSL_OP_COOKIE_EXCHANGE, SSL_OP_CRYPTOPRO_TLSEXT_BUG, SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS, SSL_OP_EPHEMERAL_RSA, SSL_OP_LEGACY_SERVER_CONNECT, SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER, SSL_OP_MICROSOFT_SESS_ID_BUG, SSL_OP_MSIE_SSLV2_RSA_PADDING, SSL_OP_NETSCAPE_CA_DN_BUG, SSL_OP_NETSCAPE_CHALLENGE_BUG, SSL_OP_NETSCAPE_DEMO_CIPHER_CHANGE_BUG, SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG, SSL_OP_NO_COMPRESSION, SSL_OP_NO_QUERY_MTU, SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION, SSL_OP_NO_SSLv2, SSL_OP_NO_SSLv3, SSL_OP_NO_TICKET, SSL_OP_NO_TLSv1, SSL_OP_NO_TLSv1_1, SSL_OP_NO_TLSv1_2, SSL_OP_PKCS1_CHECK_1, SSL_OP_PKCS1_CHECK_2, SSL_OP_SINGLE_DH_USE, SSL_OP_SINGLE_ECDH_USE, SSL_OP_SSLEAY_080_CLIENT_DH_BUG, SSL_OP_SSLREF2_REUSE_CERT_TYPE_BUG, SSL_OP_TLS_BLOCK_PADDING_BUG, SSL_OP_TLS_D5_BUG, SSL_OP_TLS_ROLLBACK_BUG, S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO, S_IFLNK, S_IFMT, S_IFREG, S_IFSOCK, S_IRGRP, S_IROTH, S_IRUSR, S_IRWXG, S_IRWXO, S_IRWXU, S_IWGRP, S_IWOTH, S_IWUSR, S_IXGRP, S_IXOTH, S_IXUSR, UV_UDP_REUSEADDR, W_OK2, X_OK2;

// node_modules/quick-format-unescaped/index.js
var require_quick_format_unescaped = __commonJS({
  "node_modules/quick-format-unescaped/index.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    function tryStringify(o9) {
      try {
        return JSON.stringify(o9);
      } catch (e10) {
        return '"[Circular]"';
      }
    }
    module.exports = format5;
    function format5(f7, args, opts) {
      var ss = opts && opts.stringify || tryStringify;
      var offset = 1;
      if (typeof f7 === "object" && f7 !== null) {
        var len = args.length + offset;
        if (len === 1)
          return f7;
        var objects = new Array(len);
        objects[0] = ss(f7);
        for (var index = 1; index < len; index++) {
          objects[index] = ss(args[index]);
        }
        return objects.join(" ");
      }
      if (typeof f7 !== "string") {
        return f7;
      }
      var argLen = args.length;
      if (argLen === 0)
        return f7;
      var str = "";
      var a7 = 1 - offset;
      var lastPos = -1;
      var flen = f7 && f7.length || 0;
      for (var i7 = 0; i7 < flen; ) {
        if (f7.charCodeAt(i7) === 37 && i7 + 1 < flen) {
          lastPos = lastPos > -1 ? lastPos : 0;
          switch (f7.charCodeAt(i7 + 1)) {
            case 100:
            case 102:
              if (a7 >= argLen)
                break;
              if (args[a7] == null)
                break;
              if (lastPos < i7)
                str += f7.slice(lastPos, i7);
              str += Number(args[a7]);
              lastPos = i7 + 2;
              i7++;
              break;
            case 105:
              if (a7 >= argLen)
                break;
              if (args[a7] == null)
                break;
              if (lastPos < i7)
                str += f7.slice(lastPos, i7);
              str += Math.floor(Number(args[a7]));
              lastPos = i7 + 2;
              i7++;
              break;
            case 79:
            case 111:
            case 106:
              if (a7 >= argLen)
                break;
              if (args[a7] === void 0)
                break;
              if (lastPos < i7)
                str += f7.slice(lastPos, i7);
              var type2 = typeof args[a7];
              if (type2 === "string") {
                str += "'" + args[a7] + "'";
                lastPos = i7 + 2;
                i7++;
                break;
              }
              if (type2 === "function") {
                str += args[a7].name || "<anonymous>";
                lastPos = i7 + 2;
                i7++;
                break;
              }
              str += ss(args[a7]);
              lastPos = i7 + 2;
              i7++;
              break;
            case 115:
              if (a7 >= argLen)
                break;
              if (lastPos < i7)
                str += f7.slice(lastPos, i7);
              str += String(args[a7]);
              lastPos = i7 + 2;
              i7++;
              break;
            case 37:
              if (lastPos < i7)
                str += f7.slice(lastPos, i7);
              str += "%";
              lastPos = i7 + 2;
              i7++;
              a7--;
              break;
          }
          ++a7;
        }
        ++i7;
      }
      if (lastPos === -1)
        return f7;
      else if (lastPos < flen) {
        str += f7.slice(lastPos);
      }
      return str;
    }
  }
});

// node_modules/pino/browser.js
var require_browser2 = __commonJS({
  "node_modules/pino/browser.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var format5 = require_quick_format_unescaped();
    module.exports = pino2;
    var _console = pfGlobalThisOrFallback().console || {};
    var stdSerializers = {
      mapHttpRequest: mock,
      mapHttpResponse: mock,
      wrapRequestSerializer: passthrough,
      wrapResponseSerializer: passthrough,
      wrapErrorSerializer: passthrough,
      req: mock,
      res: mock,
      err: asErrValue,
      errWithCause: asErrValue
    };
    function shouldSerialize(serialize, serializers) {
      if (Array.isArray(serialize)) {
        const hasToFilter = serialize.filter(function(k4) {
          return k4 !== "!stdSerializers.err";
        });
        return hasToFilter;
      } else if (serialize === true) {
        return Object.keys(serializers);
      }
      return false;
    }
    function pino2(opts) {
      opts = opts || {};
      opts.browser = opts.browser || {};
      const transmit2 = opts.browser.transmit;
      if (transmit2 && typeof transmit2.send !== "function") {
        throw Error("pino: transmit option must have a send function");
      }
      const proto = opts.browser.write || _console;
      if (opts.browser.write)
        opts.browser.asObject = true;
      const serializers = opts.serializers || {};
      const serialize = shouldSerialize(opts.browser.serialize, serializers);
      let stdErrSerialize = opts.browser.serialize;
      if (Array.isArray(opts.browser.serialize) && opts.browser.serialize.indexOf("!stdSerializers.err") > -1)
        stdErrSerialize = false;
      const customLevels = Object.keys(opts.customLevels || {});
      const levels = ["error", "fatal", "warn", "info", "debug", "trace"].concat(customLevels);
      if (typeof proto === "function") {
        levels.forEach(function(level2) {
          proto[level2] = proto;
        });
      }
      if (opts.enabled === false || opts.browser.disabled)
        opts.level = "silent";
      const level = opts.level || "info";
      const logger2 = Object.create(proto);
      if (!logger2.log)
        logger2.log = noop3;
      Object.defineProperty(logger2, "levelVal", {
        get: getLevelVal
      });
      Object.defineProperty(logger2, "level", {
        get: getLevel,
        set: setLevel
      });
      const setOpts = {
        transmit: transmit2,
        serialize,
        asObject: opts.browser.asObject,
        levels,
        timestamp: getTimeFunction(opts)
      };
      logger2.levels = getLevels(opts);
      logger2.level = level;
      logger2.setMaxListeners = logger2.getMaxListeners = logger2.emit = logger2.addListener = logger2.on = logger2.prependListener = logger2.once = logger2.prependOnceListener = logger2.removeListener = logger2.removeAllListeners = logger2.listeners = logger2.listenerCount = logger2.eventNames = logger2.write = logger2.flush = noop3;
      logger2.serializers = serializers;
      logger2._serialize = serialize;
      logger2._stdErrSerialize = stdErrSerialize;
      logger2.child = child;
      if (transmit2)
        logger2._logEvent = createLogEventShape();
      function getLevelVal() {
        return this.level === "silent" ? Infinity : this.levels.values[this.level];
      }
      function getLevel() {
        return this._level;
      }
      function setLevel(level2) {
        if (level2 !== "silent" && !this.levels.values[level2]) {
          throw Error("unknown level " + level2);
        }
        this._level = level2;
        set(setOpts, logger2, "error", "log");
        set(setOpts, logger2, "fatal", "error");
        set(setOpts, logger2, "warn", "error");
        set(setOpts, logger2, "info", "log");
        set(setOpts, logger2, "debug", "log");
        set(setOpts, logger2, "trace", "log");
        customLevels.forEach(function(level3) {
          set(setOpts, logger2, level3, "log");
        });
      }
      function child(bindings, childOptions) {
        if (!bindings) {
          throw new Error("missing bindings for child Pino");
        }
        childOptions = childOptions || {};
        if (serialize && bindings.serializers) {
          childOptions.serializers = bindings.serializers;
        }
        const childOptionsSerializers = childOptions.serializers;
        if (serialize && childOptionsSerializers) {
          var childSerializers = Object.assign({}, serializers, childOptionsSerializers);
          var childSerialize = opts.browser.serialize === true ? Object.keys(childSerializers) : serialize;
          delete bindings.serializers;
          applySerializers([bindings], childSerialize, childSerializers, this._stdErrSerialize);
        }
        function Child(parent) {
          this._childLevel = (parent._childLevel | 0) + 1;
          this.error = bind2(parent, bindings, "error");
          this.fatal = bind2(parent, bindings, "fatal");
          this.warn = bind2(parent, bindings, "warn");
          this.info = bind2(parent, bindings, "info");
          this.debug = bind2(parent, bindings, "debug");
          this.trace = bind2(parent, bindings, "trace");
          if (childSerializers) {
            this.serializers = childSerializers;
            this._serialize = childSerialize;
          }
          if (transmit2) {
            this._logEvent = createLogEventShape(
              [].concat(parent._logEvent.bindings, bindings)
            );
          }
        }
        Child.prototype = this;
        return new Child(this);
      }
      return logger2;
    }
    function getLevels(opts) {
      const customLevels = opts.customLevels || {};
      const values = Object.assign({}, pino2.levels.values, customLevels);
      const labels = Object.assign({}, pino2.levels.labels, invertObject(customLevels));
      return {
        values,
        labels
      };
    }
    function invertObject(obj) {
      const inverted = {};
      Object.keys(obj).forEach(function(key) {
        inverted[obj[key]] = key;
      });
      return inverted;
    }
    pino2.levels = {
      values: {
        fatal: 60,
        error: 50,
        warn: 40,
        info: 30,
        debug: 20,
        trace: 10
      },
      labels: {
        10: "trace",
        20: "debug",
        30: "info",
        40: "warn",
        50: "error",
        60: "fatal"
      }
    };
    pino2.stdSerializers = stdSerializers;
    pino2.stdTimeFunctions = Object.assign({}, { nullTime, epochTime, unixTime, isoTime });
    function set(opts, logger2, level, fallback) {
      const proto = Object.getPrototypeOf(logger2);
      logger2[level] = logger2.levelVal > logger2.levels.values[level] ? noop3 : proto[level] ? proto[level] : _console[level] || _console[fallback] || noop3;
      wrap(opts, logger2, level);
    }
    function wrap(opts, logger2, level) {
      if (!opts.transmit && logger2[level] === noop3)
        return;
      logger2[level] = function(write2) {
        return function LOG() {
          const ts = opts.timestamp();
          const args = new Array(arguments.length);
          const proto = Object.getPrototypeOf && Object.getPrototypeOf(this) === _console ? _console : this;
          for (var i7 = 0; i7 < args.length; i7++)
            args[i7] = arguments[i7];
          if (opts.serialize && !opts.asObject) {
            applySerializers(args, this._serialize, this.serializers, this._stdErrSerialize);
          }
          if (opts.asObject)
            write2.call(proto, asObject(this, level, args, ts));
          else
            write2.apply(proto, args);
          if (opts.transmit) {
            const transmitLevel = opts.transmit.level || logger2.level;
            const transmitValue = logger2.levels.values[transmitLevel];
            const methodValue = logger2.levels.values[level];
            if (methodValue < transmitValue)
              return;
            transmit(this, {
              ts,
              methodLevel: level,
              methodValue,
              transmitLevel,
              transmitValue: logger2.levels.values[opts.transmit.level || logger2.level],
              send: opts.transmit.send,
              val: logger2.levelVal
            }, args);
          }
        };
      }(logger2[level]);
    }
    function asObject(logger2, level, args, ts) {
      if (logger2._serialize)
        applySerializers(args, logger2._serialize, logger2.serializers, logger2._stdErrSerialize);
      const argsCloned = args.slice();
      let msg = argsCloned[0];
      const o9 = {};
      if (ts) {
        o9.time = ts;
      }
      o9.level = logger2.levels.values[level];
      let lvl = (logger2._childLevel | 0) + 1;
      if (lvl < 1)
        lvl = 1;
      if (msg !== null && typeof msg === "object") {
        while (lvl-- && typeof argsCloned[0] === "object") {
          Object.assign(o9, argsCloned.shift());
        }
        msg = argsCloned.length ? format5(argsCloned.shift(), argsCloned) : void 0;
      } else if (typeof msg === "string")
        msg = format5(argsCloned.shift(), argsCloned);
      if (msg !== void 0)
        o9.msg = msg;
      return o9;
    }
    function applySerializers(args, serialize, serializers, stdErrSerialize) {
      for (const i7 in args) {
        if (stdErrSerialize && args[i7] instanceof Error) {
          args[i7] = pino2.stdSerializers.err(args[i7]);
        } else if (typeof args[i7] === "object" && !Array.isArray(args[i7])) {
          for (const k4 in args[i7]) {
            if (serialize && serialize.indexOf(k4) > -1 && k4 in serializers) {
              args[i7][k4] = serializers[k4](args[i7][k4]);
            }
          }
        }
      }
    }
    function bind2(parent, bindings, level) {
      return function() {
        const args = new Array(1 + arguments.length);
        args[0] = bindings;
        for (var i7 = 1; i7 < args.length; i7++) {
          args[i7] = arguments[i7 - 1];
        }
        return parent[level].apply(this, args);
      };
    }
    function transmit(logger2, opts, args) {
      const send = opts.send;
      const ts = opts.ts;
      const methodLevel = opts.methodLevel;
      const methodValue = opts.methodValue;
      const val = opts.val;
      const bindings = logger2._logEvent.bindings;
      applySerializers(
        args,
        logger2._serialize || Object.keys(logger2.serializers),
        logger2.serializers,
        logger2._stdErrSerialize === void 0 ? true : logger2._stdErrSerialize
      );
      logger2._logEvent.ts = ts;
      logger2._logEvent.messages = args.filter(function(arg) {
        return bindings.indexOf(arg) === -1;
      });
      logger2._logEvent.level.label = methodLevel;
      logger2._logEvent.level.value = methodValue;
      send(methodLevel, logger2._logEvent, val);
      logger2._logEvent = createLogEventShape(bindings);
    }
    function createLogEventShape(bindings) {
      return {
        ts: 0,
        messages: [],
        bindings: bindings || [],
        level: { label: "", value: 0 }
      };
    }
    function asErrValue(err) {
      const obj = {
        type: err.constructor.name,
        msg: err.message,
        stack: err.stack
      };
      for (const key in err) {
        if (obj[key] === void 0) {
          obj[key] = err[key];
        }
      }
      return obj;
    }
    function getTimeFunction(opts) {
      if (typeof opts.timestamp === "function") {
        return opts.timestamp;
      }
      if (opts.timestamp === false) {
        return nullTime;
      }
      return epochTime;
    }
    function mock() {
      return {};
    }
    function passthrough(a7) {
      return a7;
    }
    function noop3() {
    }
    function nullTime() {
      return false;
    }
    function epochTime() {
      return Date.now();
    }
    function unixTime() {
      return Math.round(Date.now() / 1e3);
    }
    function isoTime() {
      return new Date(Date.now()).toISOString();
    }
    function pfGlobalThisOrFallback() {
      function defd(o9) {
        return typeof o9 !== "undefined" && o9;
      }
      try {
        if (typeof globalThis !== "undefined")
          return globalThis;
        Object.defineProperty(Object.prototype, "globalThis", {
          get: function() {
            delete Object.prototype.globalThis;
            return this.globalThis = this;
          },
          configurable: true
        });
        return globalThis;
      } catch (e10) {
        return defd(self) || defd(window) || defd(this) || {};
      }
    }
  }
});

// node_modules/rotating-file-stream/dist/es/index.js
var es_exports = {};
__export(es_exports, {
  RotatingFileStream: () => RotatingFileStream,
  RotatingFileStreamError: () => RotatingFileStreamError,
  createStream: () => createStream
});
function checkOpts(options) {
  const ret = {};
  for (const opt in options) {
    const value = options[opt];
    const type2 = typeof value;
    if (!(opt in checks))
      throw new Error(`Unknown option: ${opt}`);
    ret[opt] = options[opt];
    checks[opt](type2, ret, value);
  }
  if (!ret.path)
    ret.path = "";
  if (!ret.interval) {
    delete ret.immutable;
    delete ret.initialRotation;
    delete ret.intervalBoundary;
  }
  if (ret.rotate) {
    delete ret.history;
    delete ret.immutable;
    delete ret.maxFiles;
    delete ret.maxSize;
    delete ret.intervalBoundary;
  }
  if (ret.immutable)
    delete ret.compress;
  if (!ret.intervalBoundary)
    delete ret.initialRotation;
  return ret;
}
function createClassical(filename, compress, omitExtension) {
  return (index) => index ? `${filename}.${index}${compress && !omitExtension ? ".gz" : ""}` : filename;
}
function createGenerator(filename, compress, omitExtension) {
  const pad = (num) => (num > 9 ? "" : "0") + num;
  return (time, index) => {
    if (!time)
      return filename;
    const month = time.getFullYear() + "" + pad(time.getMonth() + 1);
    const day = pad(time.getDate());
    const hour = pad(time.getHours());
    const minute = pad(time.getMinutes());
    return month + day + "-" + hour + minute + "-" + pad(index) + "-" + filename + (compress && !omitExtension ? ".gz" : "");
  };
}
function createStream(filename, options) {
  if (typeof options === "undefined")
    options = {};
  else if (typeof options !== "object")
    throw new Error(`The "options" argument must be of type object. Received type ${typeof options}`);
  const opts = checkOpts(options);
  const { compress, omitExtension } = opts;
  let generator;
  if (typeof filename === "string")
    generator = options.rotate ? createClassical(filename, compress !== void 0, omitExtension) : createGenerator(filename, compress !== void 0, omitExtension);
  else if (typeof filename === "function")
    generator = filename;
  else
    throw new Error(`The "filename" argument must be one of type string or function. Received type ${typeof filename}`);
  return new RotatingFileStream(generator, opts);
}
var RotatingFileStreamError, RotatingFileStream, checks;

// node_modules/object-hash/dist/object_hash.js
var require_object_hash = __commonJS({
  "node_modules/object-hash/dist/object_hash.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    !function(e10) {
      var t9;
      "object" == typeof exports10 ? module.exports = e10() : "function" == typeof define && define.amd ? define(e10) : ("undefined" != typeof window ? t9 = window : "undefined" != typeof global ? t9 = global : "undefined" != typeof self && (t9 = self), t9.objectHash = e10());
    }(function() {
      return function r10(o9, i7, u7) {
        function s6(n9, e11) {
          if (!i7[n9]) {
            if (!o9[n9]) {
              var t9 = "function" == typeof __require && __require;
              if (!e11 && t9)
                return t9(n9, true);
              if (a7)
                return a7(n9, true);
              throw new Error("Cannot find module '" + n9 + "'");
            }
            e11 = i7[n9] = { exports: {} };
            o9[n9][0].call(e11.exports, function(e12) {
              var t10 = o9[n9][1][e12];
              return s6(t10 || e12);
            }, e11, e11.exports, r10, o9, i7, u7);
          }
          return i7[n9].exports;
        }
        for (var a7 = "function" == typeof __require && __require, e10 = 0; e10 < u7.length; e10++)
          s6(u7[e10]);
        return s6;
      }({ 1: [function(w4, b5, m6) {
        !function(e10, n9, s6, c7, d6, h8, p7, g5, y6) {
          var r10 = w4("crypto");
          function t9(e11, t10) {
            t10 = u7(e11, t10);
            var n10;
            return void 0 === (n10 = "passthrough" !== t10.algorithm ? r10.createHash(t10.algorithm) : new l7()).write && (n10.write = n10.update, n10.end = n10.update), f7(t10, n10).dispatch(e11), n10.update || n10.end(""), n10.digest ? n10.digest("buffer" === t10.encoding ? void 0 : t10.encoding) : (e11 = n10.read(), "buffer" !== t10.encoding ? e11.toString(t10.encoding) : e11);
          }
          (m6 = b5.exports = t9).sha1 = function(e11) {
            return t9(e11);
          }, m6.keys = function(e11) {
            return t9(e11, { excludeValues: true, algorithm: "sha1", encoding: "hex" });
          }, m6.MD5 = function(e11) {
            return t9(e11, { algorithm: "md5", encoding: "hex" });
          }, m6.keysMD5 = function(e11) {
            return t9(e11, { algorithm: "md5", encoding: "hex", excludeValues: true });
          };
          var o9 = r10.getHashes ? r10.getHashes().slice() : ["sha1", "md5"], i7 = (o9.push("passthrough"), ["buffer", "hex", "binary", "base64"]);
          function u7(e11, t10) {
            var n10 = {};
            if (n10.algorithm = (t10 = t10 || {}).algorithm || "sha1", n10.encoding = t10.encoding || "hex", n10.excludeValues = !!t10.excludeValues, n10.algorithm = n10.algorithm.toLowerCase(), n10.encoding = n10.encoding.toLowerCase(), n10.ignoreUnknown = true === t10.ignoreUnknown, n10.respectType = false !== t10.respectType, n10.respectFunctionNames = false !== t10.respectFunctionNames, n10.respectFunctionProperties = false !== t10.respectFunctionProperties, n10.unorderedArrays = true === t10.unorderedArrays, n10.unorderedSets = false !== t10.unorderedSets, n10.unorderedObjects = false !== t10.unorderedObjects, n10.replacer = t10.replacer || void 0, n10.excludeKeys = t10.excludeKeys || void 0, void 0 === e11)
              throw new Error("Object argument required.");
            for (var r11 = 0; r11 < o9.length; ++r11)
              o9[r11].toLowerCase() === n10.algorithm.toLowerCase() && (n10.algorithm = o9[r11]);
            if (-1 === o9.indexOf(n10.algorithm))
              throw new Error('Algorithm "' + n10.algorithm + '"  not supported. supported values: ' + o9.join(", "));
            if (-1 === i7.indexOf(n10.encoding) && "passthrough" !== n10.algorithm)
              throw new Error('Encoding "' + n10.encoding + '"  not supported. supported values: ' + i7.join(", "));
            return n10;
          }
          function a7(e11) {
            if ("function" == typeof e11)
              return null != /^function\s+\w*\s*\(\s*\)\s*{\s+\[native code\]\s+}$/i.exec(Function.prototype.toString.call(e11));
          }
          function f7(o10, t10, i8) {
            i8 = i8 || [];
            function u8(e11) {
              return t10.update ? t10.update(e11, "utf8") : t10.write(e11, "utf8");
            }
            return { dispatch: function(e11) {
              return this["_" + (null === (e11 = o10.replacer ? o10.replacer(e11) : e11) ? "null" : typeof e11)](e11);
            }, _object: function(t11) {
              var n10, e11 = Object.prototype.toString.call(t11), r11 = /\[object (.*)\]/i.exec(e11);
              r11 = (r11 = r11 ? r11[1] : "unknown:[" + e11 + "]").toLowerCase();
              if (0 <= (e11 = i8.indexOf(t11)))
                return this.dispatch("[CIRCULAR:" + e11 + "]");
              if (i8.push(t11), void 0 !== s6 && s6.isBuffer && s6.isBuffer(t11))
                return u8("buffer:"), u8(t11);
              if ("object" === r11 || "function" === r11 || "asyncfunction" === r11)
                return e11 = Object.keys(t11), o10.unorderedObjects && (e11 = e11.sort()), false === o10.respectType || a7(t11) || e11.splice(0, 0, "prototype", "__proto__", "constructor"), o10.excludeKeys && (e11 = e11.filter(function(e12) {
                  return !o10.excludeKeys(e12);
                })), u8("object:" + e11.length + ":"), n10 = this, e11.forEach(function(e12) {
                  n10.dispatch(e12), u8(":"), o10.excludeValues || n10.dispatch(t11[e12]), u8(",");
                });
              if (!this["_" + r11]) {
                if (o10.ignoreUnknown)
                  return u8("[" + r11 + "]");
                throw new Error('Unknown object type "' + r11 + '"');
              }
              this["_" + r11](t11);
            }, _array: function(e11, t11) {
              t11 = void 0 !== t11 ? t11 : false !== o10.unorderedArrays;
              var n10 = this;
              if (u8("array:" + e11.length + ":"), !t11 || e11.length <= 1)
                return e11.forEach(function(e12) {
                  return n10.dispatch(e12);
                });
              var r11 = [], t11 = e11.map(function(e12) {
                var t12 = new l7(), n11 = i8.slice();
                return f7(o10, t12, n11).dispatch(e12), r11 = r11.concat(n11.slice(i8.length)), t12.read().toString();
              });
              return i8 = i8.concat(r11), t11.sort(), this._array(t11, false);
            }, _date: function(e11) {
              return u8("date:" + e11.toJSON());
            }, _symbol: function(e11) {
              return u8("symbol:" + e11.toString());
            }, _error: function(e11) {
              return u8("error:" + e11.toString());
            }, _boolean: function(e11) {
              return u8("bool:" + e11.toString());
            }, _string: function(e11) {
              u8("string:" + e11.length + ":"), u8(e11.toString());
            }, _function: function(e11) {
              u8("fn:"), a7(e11) ? this.dispatch("[native]") : this.dispatch(e11.toString()), false !== o10.respectFunctionNames && this.dispatch("function-name:" + String(e11.name)), o10.respectFunctionProperties && this._object(e11);
            }, _number: function(e11) {
              return u8("number:" + e11.toString());
            }, _xml: function(e11) {
              return u8("xml:" + e11.toString());
            }, _null: function() {
              return u8("Null");
            }, _undefined: function() {
              return u8("Undefined");
            }, _regexp: function(e11) {
              return u8("regex:" + e11.toString());
            }, _uint8array: function(e11) {
              return u8("uint8array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _uint8clampedarray: function(e11) {
              return u8("uint8clampedarray:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _int8array: function(e11) {
              return u8("int8array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _uint16array: function(e11) {
              return u8("uint16array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _int16array: function(e11) {
              return u8("int16array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _uint32array: function(e11) {
              return u8("uint32array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _int32array: function(e11) {
              return u8("int32array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _float32array: function(e11) {
              return u8("float32array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _float64array: function(e11) {
              return u8("float64array:"), this.dispatch(Array.prototype.slice.call(e11));
            }, _arraybuffer: function(e11) {
              return u8("arraybuffer:"), this.dispatch(new Uint8Array(e11));
            }, _url: function(e11) {
              return u8("url:" + e11.toString());
            }, _map: function(e11) {
              u8("map:");
              e11 = Array.from(e11);
              return this._array(e11, false !== o10.unorderedSets);
            }, _set: function(e11) {
              u8("set:");
              e11 = Array.from(e11);
              return this._array(e11, false !== o10.unorderedSets);
            }, _file: function(e11) {
              return u8("file:"), this.dispatch([e11.name, e11.size, e11.type, e11.lastModfied]);
            }, _blob: function() {
              if (o10.ignoreUnknown)
                return u8("[blob]");
              throw Error('Hashing Blob objects is currently not supported\n(see https://github.com/puleos/object-hash/issues/26)\nUse "options.replacer" or "options.ignoreUnknown"\n');
            }, _domwindow: function() {
              return u8("domwindow");
            }, _bigint: function(e11) {
              return u8("bigint:" + e11.toString());
            }, _process: function() {
              return u8("process");
            }, _timer: function() {
              return u8("timer");
            }, _pipe: function() {
              return u8("pipe");
            }, _tcp: function() {
              return u8("tcp");
            }, _udp: function() {
              return u8("udp");
            }, _tty: function() {
              return u8("tty");
            }, _statwatcher: function() {
              return u8("statwatcher");
            }, _securecontext: function() {
              return u8("securecontext");
            }, _connection: function() {
              return u8("connection");
            }, _zlib: function() {
              return u8("zlib");
            }, _context: function() {
              return u8("context");
            }, _nodescript: function() {
              return u8("nodescript");
            }, _httpparser: function() {
              return u8("httpparser");
            }, _dataview: function() {
              return u8("dataview");
            }, _signal: function() {
              return u8("signal");
            }, _fsevent: function() {
              return u8("fsevent");
            }, _tlswrap: function() {
              return u8("tlswrap");
            } };
          }
          function l7() {
            return { buf: "", write: function(e11) {
              this.buf += e11;
            }, end: function(e11) {
              this.buf += e11;
            }, read: function() {
              return this.buf;
            } };
          }
          m6.writeToStream = function(e11, t10, n10) {
            return void 0 === n10 && (n10 = t10, t10 = {}), f7(t10 = u7(e11, t10), n10).dispatch(e11);
          };
        }.call(this, w4("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, w4("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/fake_9a5aa49d.js", "/");
      }, { buffer: 3, crypto: 5, lYpoI2: 11 }], 2: [function(e10, t9, f7) {
        !function(e11, t10, n9, r10, o9, i7, u7, s6, a7) {
          !function(e12) {
            var a8 = "undefined" != typeof Uint8Array ? Uint8Array : Array, t11 = "+".charCodeAt(0), n10 = "/".charCodeAt(0), r11 = "0".charCodeAt(0), o10 = "a".charCodeAt(0), i8 = "A".charCodeAt(0), u8 = "-".charCodeAt(0), s7 = "_".charCodeAt(0);
            function f8(e13) {
              e13 = e13.charCodeAt(0);
              return e13 === t11 || e13 === u8 ? 62 : e13 === n10 || e13 === s7 ? 63 : e13 < r11 ? -1 : e13 < r11 + 10 ? e13 - r11 + 26 + 26 : e13 < i8 + 26 ? e13 - i8 : e13 < o10 + 26 ? e13 - o10 + 26 : void 0;
            }
            e12.toByteArray = function(e13) {
              var t12, n11;
              if (0 < e13.length % 4)
                throw new Error("Invalid string. Length must be a multiple of 4");
              var r12 = e13.length, r12 = "=" === e13.charAt(r12 - 2) ? 2 : "=" === e13.charAt(r12 - 1) ? 1 : 0, o11 = new a8(3 * e13.length / 4 - r12), i9 = 0 < r12 ? e13.length - 4 : e13.length, u9 = 0;
              function s8(e14) {
                o11[u9++] = e14;
              }
              for (t12 = 0; t12 < i9; t12 += 4, 0)
                s8((16711680 & (n11 = f8(e13.charAt(t12)) << 18 | f8(e13.charAt(t12 + 1)) << 12 | f8(e13.charAt(t12 + 2)) << 6 | f8(e13.charAt(t12 + 3)))) >> 16), s8((65280 & n11) >> 8), s8(255 & n11);
              return 2 == r12 ? s8(255 & (n11 = f8(e13.charAt(t12)) << 2 | f8(e13.charAt(t12 + 1)) >> 4)) : 1 == r12 && (s8((n11 = f8(e13.charAt(t12)) << 10 | f8(e13.charAt(t12 + 1)) << 4 | f8(e13.charAt(t12 + 2)) >> 2) >> 8 & 255), s8(255 & n11)), o11;
            }, e12.fromByteArray = function(e13) {
              var t12, n11, r12, o11, i9 = e13.length % 3, u9 = "";
              function s8(e14) {
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(e14);
              }
              for (t12 = 0, r12 = e13.length - i9; t12 < r12; t12 += 3)
                n11 = (e13[t12] << 16) + (e13[t12 + 1] << 8) + e13[t12 + 2], u9 += s8((o11 = n11) >> 18 & 63) + s8(o11 >> 12 & 63) + s8(o11 >> 6 & 63) + s8(63 & o11);
              switch (i9) {
                case 1:
                  u9 = (u9 += s8((n11 = e13[e13.length - 1]) >> 2)) + s8(n11 << 4 & 63) + "==";
                  break;
                case 2:
                  u9 = (u9 = (u9 += s8((n11 = (e13[e13.length - 2] << 8) + e13[e13.length - 1]) >> 10)) + s8(n11 >> 4 & 63)) + s8(n11 << 2 & 63) + "=";
              }
              return u9;
            };
          }(void 0 === f7 ? this.base64js = {} : f7);
        }.call(this, e10("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e10("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/base64-js/lib/b64.js", "/node_modules/gulp-browserify/node_modules/base64-js/lib");
      }, { buffer: 3, lYpoI2: 11 }], 3: [function(O5, e10, H3) {
        !function(e11, n9, f7, r10, h8, p7, g5, y6, w4) {
          var a7 = O5("base64-js"), i7 = O5("ieee754");
          function f7(e12, t10, n10) {
            if (!(this instanceof f7))
              return new f7(e12, t10, n10);
            var r11, o10, i8, u8, s7 = typeof e12;
            if ("base64" === t10 && "string" == s7)
              for (e12 = (u8 = e12).trim ? u8.trim() : u8.replace(/^\s+|\s+$/g, ""); e12.length % 4 != 0; )
                e12 += "=";
            if ("number" == s7)
              r11 = j4(e12);
            else if ("string" == s7)
              r11 = f7.byteLength(e12, t10);
            else {
              if ("object" != s7)
                throw new Error("First argument needs to be a number, array or string.");
              r11 = j4(e12.length);
            }
            if (f7._useTypedArrays ? o10 = f7._augment(new Uint8Array(r11)) : ((o10 = this).length = r11, o10._isBuffer = true), f7._useTypedArrays && "number" == typeof e12.byteLength)
              o10._set(e12);
            else if (C4(u8 = e12) || f7.isBuffer(u8) || u8 && "object" == typeof u8 && "number" == typeof u8.length)
              for (i8 = 0; i8 < r11; i8++)
                f7.isBuffer(e12) ? o10[i8] = e12.readUInt8(i8) : o10[i8] = e12[i8];
            else if ("string" == s7)
              o10.write(e12, 0, t10);
            else if ("number" == s7 && !f7._useTypedArrays && !n10)
              for (i8 = 0; i8 < r11; i8++)
                o10[i8] = 0;
            return o10;
          }
          function b5(e12, t10, n10, r11) {
            return f7._charsWritten = c7(function(e13) {
              for (var t11 = [], n11 = 0; n11 < e13.length; n11++)
                t11.push(255 & e13.charCodeAt(n11));
              return t11;
            }(t10), e12, n10, r11);
          }
          function m6(e12, t10, n10, r11) {
            return f7._charsWritten = c7(function(e13) {
              for (var t11, n11, r12 = [], o10 = 0; o10 < e13.length; o10++)
                n11 = e13.charCodeAt(o10), t11 = n11 >> 8, n11 = n11 % 256, r12.push(n11), r12.push(t11);
              return r12;
            }(t10), e12, n10, r11);
          }
          function v7(e12, t10, n10) {
            var r11 = "";
            n10 = Math.min(e12.length, n10);
            for (var o10 = t10; o10 < n10; o10++)
              r11 += String.fromCharCode(e12[o10]);
            return r11;
          }
          function o9(e12, t10, n10, r11) {
            r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(null != t10, "missing offset"), d6(t10 + 1 < e12.length, "Trying to read beyond buffer length"));
            var o10, r11 = e12.length;
            if (!(r11 <= t10))
              return n10 ? (o10 = e12[t10], t10 + 1 < r11 && (o10 |= e12[t10 + 1] << 8)) : (o10 = e12[t10] << 8, t10 + 1 < r11 && (o10 |= e12[t10 + 1])), o10;
          }
          function u7(e12, t10, n10, r11) {
            r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(null != t10, "missing offset"), d6(t10 + 3 < e12.length, "Trying to read beyond buffer length"));
            var o10, r11 = e12.length;
            if (!(r11 <= t10))
              return n10 ? (t10 + 2 < r11 && (o10 = e12[t10 + 2] << 16), t10 + 1 < r11 && (o10 |= e12[t10 + 1] << 8), o10 |= e12[t10], t10 + 3 < r11 && (o10 += e12[t10 + 3] << 24 >>> 0)) : (t10 + 1 < r11 && (o10 = e12[t10 + 1] << 16), t10 + 2 < r11 && (o10 |= e12[t10 + 2] << 8), t10 + 3 < r11 && (o10 |= e12[t10 + 3]), o10 += e12[t10] << 24 >>> 0), o10;
          }
          function _4(e12, t10, n10, r11) {
            if (r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(null != t10, "missing offset"), d6(t10 + 1 < e12.length, "Trying to read beyond buffer length")), !(e12.length <= t10))
              return r11 = o9(e12, t10, n10, true), 32768 & r11 ? -1 * (65535 - r11 + 1) : r11;
          }
          function E4(e12, t10, n10, r11) {
            if (r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(null != t10, "missing offset"), d6(t10 + 3 < e12.length, "Trying to read beyond buffer length")), !(e12.length <= t10))
              return r11 = u7(e12, t10, n10, true), 2147483648 & r11 ? -1 * (4294967295 - r11 + 1) : r11;
          }
          function I4(e12, t10, n10, r11) {
            return r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(t10 + 3 < e12.length, "Trying to read beyond buffer length")), i7.read(e12, t10, n10, 23, 4);
          }
          function A4(e12, t10, n10, r11) {
            return r11 || (d6("boolean" == typeof n10, "missing or invalid endian"), d6(t10 + 7 < e12.length, "Trying to read beyond buffer length")), i7.read(e12, t10, n10, 52, 8);
          }
          function s6(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 1 < e12.length, "trying to write beyond buffer length"), Y4(t10, 65535));
            o10 = e12.length;
            if (!(o10 <= n10))
              for (var i8 = 0, u8 = Math.min(o10 - n10, 2); i8 < u8; i8++)
                e12[n10 + i8] = (t10 & 255 << 8 * (r11 ? i8 : 1 - i8)) >>> 8 * (r11 ? i8 : 1 - i8);
          }
          function l7(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 3 < e12.length, "trying to write beyond buffer length"), Y4(t10, 4294967295));
            o10 = e12.length;
            if (!(o10 <= n10))
              for (var i8 = 0, u8 = Math.min(o10 - n10, 4); i8 < u8; i8++)
                e12[n10 + i8] = t10 >>> 8 * (r11 ? i8 : 3 - i8) & 255;
          }
          function B4(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 1 < e12.length, "Trying to write beyond buffer length"), F4(t10, 32767, -32768)), e12.length <= n10 || s6(e12, 0 <= t10 ? t10 : 65535 + t10 + 1, n10, r11, o10);
          }
          function L4(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 3 < e12.length, "Trying to write beyond buffer length"), F4(t10, 2147483647, -2147483648)), e12.length <= n10 || l7(e12, 0 <= t10 ? t10 : 4294967295 + t10 + 1, n10, r11, o10);
          }
          function U4(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 3 < e12.length, "Trying to write beyond buffer length"), D4(t10, 34028234663852886e22, -34028234663852886e22)), e12.length <= n10 || i7.write(e12, t10, n10, r11, 23, 4);
          }
          function x4(e12, t10, n10, r11, o10) {
            o10 || (d6(null != t10, "missing value"), d6("boolean" == typeof r11, "missing or invalid endian"), d6(null != n10, "missing offset"), d6(n10 + 7 < e12.length, "Trying to write beyond buffer length"), D4(t10, 17976931348623157e292, -17976931348623157e292)), e12.length <= n10 || i7.write(e12, t10, n10, r11, 52, 8);
          }
          H3.Buffer = f7, H3.SlowBuffer = f7, H3.INSPECT_MAX_BYTES = 50, f7.poolSize = 8192, f7._useTypedArrays = function() {
            try {
              var e12 = new ArrayBuffer(0), t10 = new Uint8Array(e12);
              return t10.foo = function() {
                return 42;
              }, 42 === t10.foo() && "function" == typeof t10.subarray;
            } catch (e13) {
              return false;
            }
          }(), f7.isEncoding = function(e12) {
            switch (String(e12).toLowerCase()) {
              case "hex":
              case "utf8":
              case "utf-8":
              case "ascii":
              case "binary":
              case "base64":
              case "raw":
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                return true;
              default:
                return false;
            }
          }, f7.isBuffer = function(e12) {
            return !(null == e12 || !e12._isBuffer);
          }, f7.byteLength = function(e12, t10) {
            var n10;
            switch (e12 += "", t10 || "utf8") {
              case "hex":
                n10 = e12.length / 2;
                break;
              case "utf8":
              case "utf-8":
                n10 = T5(e12).length;
                break;
              case "ascii":
              case "binary":
              case "raw":
                n10 = e12.length;
                break;
              case "base64":
                n10 = M4(e12).length;
                break;
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                n10 = 2 * e12.length;
                break;
              default:
                throw new Error("Unknown encoding");
            }
            return n10;
          }, f7.concat = function(e12, t10) {
            if (d6(C4(e12), "Usage: Buffer.concat(list, [totalLength])\nlist should be an Array."), 0 === e12.length)
              return new f7(0);
            if (1 === e12.length)
              return e12[0];
            if ("number" != typeof t10)
              for (o10 = t10 = 0; o10 < e12.length; o10++)
                t10 += e12[o10].length;
            for (var n10 = new f7(t10), r11 = 0, o10 = 0; o10 < e12.length; o10++) {
              var i8 = e12[o10];
              i8.copy(n10, r11), r11 += i8.length;
            }
            return n10;
          }, f7.prototype.write = function(e12, t10, n10, r11) {
            isFinite(t10) ? isFinite(n10) || (r11 = n10, n10 = void 0) : (a8 = r11, r11 = t10, t10 = n10, n10 = a8), t10 = Number(t10) || 0;
            var o10, i8, u8, s7, a8 = this.length - t10;
            switch ((!n10 || a8 < (n10 = Number(n10))) && (n10 = a8), r11 = String(r11 || "utf8").toLowerCase()) {
              case "hex":
                o10 = function(e13, t11, n11, r12) {
                  n11 = Number(n11) || 0;
                  var o11 = e13.length - n11;
                  (!r12 || o11 < (r12 = Number(r12))) && (r12 = o11), d6((o11 = t11.length) % 2 == 0, "Invalid hex string"), o11 / 2 < r12 && (r12 = o11 / 2);
                  for (var i9 = 0; i9 < r12; i9++) {
                    var u9 = parseInt(t11.substr(2 * i9, 2), 16);
                    d6(!isNaN(u9), "Invalid hex string"), e13[n11 + i9] = u9;
                  }
                  return f7._charsWritten = 2 * i9, i9;
                }(this, e12, t10, n10);
                break;
              case "utf8":
              case "utf-8":
                i8 = this, u8 = t10, s7 = n10, o10 = f7._charsWritten = c7(T5(e12), i8, u8, s7);
                break;
              case "ascii":
              case "binary":
                o10 = b5(this, e12, t10, n10);
                break;
              case "base64":
                i8 = this, u8 = t10, s7 = n10, o10 = f7._charsWritten = c7(M4(e12), i8, u8, s7);
                break;
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                o10 = m6(this, e12, t10, n10);
                break;
              default:
                throw new Error("Unknown encoding");
            }
            return o10;
          }, f7.prototype.toString = function(e12, t10, n10) {
            var r11, o10, i8, u8, s7 = this;
            if (e12 = String(e12 || "utf8").toLowerCase(), t10 = Number(t10) || 0, (n10 = void 0 !== n10 ? Number(n10) : s7.length) === t10)
              return "";
            switch (e12) {
              case "hex":
                r11 = function(e13, t11, n11) {
                  var r12 = e13.length;
                  (!t11 || t11 < 0) && (t11 = 0);
                  (!n11 || n11 < 0 || r12 < n11) && (n11 = r12);
                  for (var o11 = "", i9 = t11; i9 < n11; i9++)
                    o11 += k4(e13[i9]);
                  return o11;
                }(s7, t10, n10);
                break;
              case "utf8":
              case "utf-8":
                r11 = function(e13, t11, n11) {
                  var r12 = "", o11 = "";
                  n11 = Math.min(e13.length, n11);
                  for (var i9 = t11; i9 < n11; i9++)
                    e13[i9] <= 127 ? (r12 += N4(o11) + String.fromCharCode(e13[i9]), o11 = "") : o11 += "%" + e13[i9].toString(16);
                  return r12 + N4(o11);
                }(s7, t10, n10);
                break;
              case "ascii":
              case "binary":
                r11 = v7(s7, t10, n10);
                break;
              case "base64":
                o10 = s7, u8 = n10, r11 = 0 === (i8 = t10) && u8 === o10.length ? a7.fromByteArray(o10) : a7.fromByteArray(o10.slice(i8, u8));
                break;
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                r11 = function(e13, t11, n11) {
                  for (var r12 = e13.slice(t11, n11), o11 = "", i9 = 0; i9 < r12.length; i9 += 2)
                    o11 += String.fromCharCode(r12[i9] + 256 * r12[i9 + 1]);
                  return o11;
                }(s7, t10, n10);
                break;
              default:
                throw new Error("Unknown encoding");
            }
            return r11;
          }, f7.prototype.toJSON = function() {
            return { type: "Buffer", data: Array.prototype.slice.call(this._arr || this, 0) };
          }, f7.prototype.copy = function(e12, t10, n10, r11) {
            if (t10 = t10 || 0, (r11 = r11 || 0 === r11 ? r11 : this.length) !== (n10 = n10 || 0) && 0 !== e12.length && 0 !== this.length) {
              d6(n10 <= r11, "sourceEnd < sourceStart"), d6(0 <= t10 && t10 < e12.length, "targetStart out of bounds"), d6(0 <= n10 && n10 < this.length, "sourceStart out of bounds"), d6(0 <= r11 && r11 <= this.length, "sourceEnd out of bounds"), r11 > this.length && (r11 = this.length);
              var o10 = (r11 = e12.length - t10 < r11 - n10 ? e12.length - t10 + n10 : r11) - n10;
              if (o10 < 100 || !f7._useTypedArrays)
                for (var i8 = 0; i8 < o10; i8++)
                  e12[i8 + t10] = this[i8 + n10];
              else
                e12._set(this.subarray(n10, n10 + o10), t10);
            }
          }, f7.prototype.slice = function(e12, t10) {
            var n10 = this.length;
            if (e12 = S4(e12, n10, 0), t10 = S4(t10, n10, n10), f7._useTypedArrays)
              return f7._augment(this.subarray(e12, t10));
            for (var r11 = t10 - e12, o10 = new f7(r11, void 0, true), i8 = 0; i8 < r11; i8++)
              o10[i8] = this[i8 + e12];
            return o10;
          }, f7.prototype.get = function(e12) {
            return console.log(".get() is deprecated. Access using array indexes instead."), this.readUInt8(e12);
          }, f7.prototype.set = function(e12, t10) {
            return console.log(".set() is deprecated. Access using array indexes instead."), this.writeUInt8(e12, t10);
          }, f7.prototype.readUInt8 = function(e12, t10) {
            if (t10 || (d6(null != e12, "missing offset"), d6(e12 < this.length, "Trying to read beyond buffer length")), !(e12 >= this.length))
              return this[e12];
          }, f7.prototype.readUInt16LE = function(e12, t10) {
            return o9(this, e12, true, t10);
          }, f7.prototype.readUInt16BE = function(e12, t10) {
            return o9(this, e12, false, t10);
          }, f7.prototype.readUInt32LE = function(e12, t10) {
            return u7(this, e12, true, t10);
          }, f7.prototype.readUInt32BE = function(e12, t10) {
            return u7(this, e12, false, t10);
          }, f7.prototype.readInt8 = function(e12, t10) {
            if (t10 || (d6(null != e12, "missing offset"), d6(e12 < this.length, "Trying to read beyond buffer length")), !(e12 >= this.length))
              return 128 & this[e12] ? -1 * (255 - this[e12] + 1) : this[e12];
          }, f7.prototype.readInt16LE = function(e12, t10) {
            return _4(this, e12, true, t10);
          }, f7.prototype.readInt16BE = function(e12, t10) {
            return _4(this, e12, false, t10);
          }, f7.prototype.readInt32LE = function(e12, t10) {
            return E4(this, e12, true, t10);
          }, f7.prototype.readInt32BE = function(e12, t10) {
            return E4(this, e12, false, t10);
          }, f7.prototype.readFloatLE = function(e12, t10) {
            return I4(this, e12, true, t10);
          }, f7.prototype.readFloatBE = function(e12, t10) {
            return I4(this, e12, false, t10);
          }, f7.prototype.readDoubleLE = function(e12, t10) {
            return A4(this, e12, true, t10);
          }, f7.prototype.readDoubleBE = function(e12, t10) {
            return A4(this, e12, false, t10);
          }, f7.prototype.writeUInt8 = function(e12, t10, n10) {
            n10 || (d6(null != e12, "missing value"), d6(null != t10, "missing offset"), d6(t10 < this.length, "trying to write beyond buffer length"), Y4(e12, 255)), t10 >= this.length || (this[t10] = e12);
          }, f7.prototype.writeUInt16LE = function(e12, t10, n10) {
            s6(this, e12, t10, true, n10);
          }, f7.prototype.writeUInt16BE = function(e12, t10, n10) {
            s6(this, e12, t10, false, n10);
          }, f7.prototype.writeUInt32LE = function(e12, t10, n10) {
            l7(this, e12, t10, true, n10);
          }, f7.prototype.writeUInt32BE = function(e12, t10, n10) {
            l7(this, e12, t10, false, n10);
          }, f7.prototype.writeInt8 = function(e12, t10, n10) {
            n10 || (d6(null != e12, "missing value"), d6(null != t10, "missing offset"), d6(t10 < this.length, "Trying to write beyond buffer length"), F4(e12, 127, -128)), t10 >= this.length || (0 <= e12 ? this.writeUInt8(e12, t10, n10) : this.writeUInt8(255 + e12 + 1, t10, n10));
          }, f7.prototype.writeInt16LE = function(e12, t10, n10) {
            B4(this, e12, t10, true, n10);
          }, f7.prototype.writeInt16BE = function(e12, t10, n10) {
            B4(this, e12, t10, false, n10);
          }, f7.prototype.writeInt32LE = function(e12, t10, n10) {
            L4(this, e12, t10, true, n10);
          }, f7.prototype.writeInt32BE = function(e12, t10, n10) {
            L4(this, e12, t10, false, n10);
          }, f7.prototype.writeFloatLE = function(e12, t10, n10) {
            U4(this, e12, t10, true, n10);
          }, f7.prototype.writeFloatBE = function(e12, t10, n10) {
            U4(this, e12, t10, false, n10);
          }, f7.prototype.writeDoubleLE = function(e12, t10, n10) {
            x4(this, e12, t10, true, n10);
          }, f7.prototype.writeDoubleBE = function(e12, t10, n10) {
            x4(this, e12, t10, false, n10);
          }, f7.prototype.fill = function(e12, t10, n10) {
            if (t10 = t10 || 0, n10 = n10 || this.length, d6("number" == typeof (e12 = "string" == typeof (e12 = e12 || 0) ? e12.charCodeAt(0) : e12) && !isNaN(e12), "value is not a number"), d6(t10 <= n10, "end < start"), n10 !== t10 && 0 !== this.length) {
              d6(0 <= t10 && t10 < this.length, "start out of bounds"), d6(0 <= n10 && n10 <= this.length, "end out of bounds");
              for (var r11 = t10; r11 < n10; r11++)
                this[r11] = e12;
            }
          }, f7.prototype.inspect = function() {
            for (var e12 = [], t10 = this.length, n10 = 0; n10 < t10; n10++)
              if (e12[n10] = k4(this[n10]), n10 === H3.INSPECT_MAX_BYTES) {
                e12[n10 + 1] = "...";
                break;
              }
            return "<Buffer " + e12.join(" ") + ">";
          }, f7.prototype.toArrayBuffer = function() {
            if ("undefined" == typeof Uint8Array)
              throw new Error("Buffer.toArrayBuffer not supported in this browser");
            if (f7._useTypedArrays)
              return new f7(this).buffer;
            for (var e12 = new Uint8Array(this.length), t10 = 0, n10 = e12.length; t10 < n10; t10 += 1)
              e12[t10] = this[t10];
            return e12.buffer;
          };
          var t9 = f7.prototype;
          function S4(e12, t10, n10) {
            return "number" != typeof e12 ? n10 : t10 <= (e12 = ~~e12) ? t10 : 0 <= e12 || 0 <= (e12 += t10) ? e12 : 0;
          }
          function j4(e12) {
            return (e12 = ~~Math.ceil(+e12)) < 0 ? 0 : e12;
          }
          function C4(e12) {
            return (Array.isArray || function(e13) {
              return "[object Array]" === Object.prototype.toString.call(e13);
            })(e12);
          }
          function k4(e12) {
            return e12 < 16 ? "0" + e12.toString(16) : e12.toString(16);
          }
          function T5(e12) {
            for (var t10 = [], n10 = 0; n10 < e12.length; n10++) {
              var r11 = e12.charCodeAt(n10);
              if (r11 <= 127)
                t10.push(e12.charCodeAt(n10));
              else
                for (var o10 = n10, i8 = (55296 <= r11 && r11 <= 57343 && n10++, encodeURIComponent(e12.slice(o10, n10 + 1)).substr(1).split("%")), u8 = 0; u8 < i8.length; u8++)
                  t10.push(parseInt(i8[u8], 16));
            }
            return t10;
          }
          function M4(e12) {
            return a7.toByteArray(e12);
          }
          function c7(e12, t10, n10, r11) {
            for (var o10 = 0; o10 < r11 && !(o10 + n10 >= t10.length || o10 >= e12.length); o10++)
              t10[o10 + n10] = e12[o10];
            return o10;
          }
          function N4(e12) {
            try {
              return decodeURIComponent(e12);
            } catch (e13) {
              return String.fromCharCode(65533);
            }
          }
          function Y4(e12, t10) {
            d6("number" == typeof e12, "cannot write a non-number as a number"), d6(0 <= e12, "specified a negative value for writing an unsigned value"), d6(e12 <= t10, "value is larger than maximum value for type"), d6(Math.floor(e12) === e12, "value has a fractional component");
          }
          function F4(e12, t10, n10) {
            d6("number" == typeof e12, "cannot write a non-number as a number"), d6(e12 <= t10, "value larger than maximum allowed value"), d6(n10 <= e12, "value smaller than minimum allowed value"), d6(Math.floor(e12) === e12, "value has a fractional component");
          }
          function D4(e12, t10, n10) {
            d6("number" == typeof e12, "cannot write a non-number as a number"), d6(e12 <= t10, "value larger than maximum allowed value"), d6(n10 <= e12, "value smaller than minimum allowed value");
          }
          function d6(e12, t10) {
            if (!e12)
              throw new Error(t10 || "Failed assertion");
          }
          f7._augment = function(e12) {
            return e12._isBuffer = true, e12._get = e12.get, e12._set = e12.set, e12.get = t9.get, e12.set = t9.set, e12.write = t9.write, e12.toString = t9.toString, e12.toLocaleString = t9.toString, e12.toJSON = t9.toJSON, e12.copy = t9.copy, e12.slice = t9.slice, e12.readUInt8 = t9.readUInt8, e12.readUInt16LE = t9.readUInt16LE, e12.readUInt16BE = t9.readUInt16BE, e12.readUInt32LE = t9.readUInt32LE, e12.readUInt32BE = t9.readUInt32BE, e12.readInt8 = t9.readInt8, e12.readInt16LE = t9.readInt16LE, e12.readInt16BE = t9.readInt16BE, e12.readInt32LE = t9.readInt32LE, e12.readInt32BE = t9.readInt32BE, e12.readFloatLE = t9.readFloatLE, e12.readFloatBE = t9.readFloatBE, e12.readDoubleLE = t9.readDoubleLE, e12.readDoubleBE = t9.readDoubleBE, e12.writeUInt8 = t9.writeUInt8, e12.writeUInt16LE = t9.writeUInt16LE, e12.writeUInt16BE = t9.writeUInt16BE, e12.writeUInt32LE = t9.writeUInt32LE, e12.writeUInt32BE = t9.writeUInt32BE, e12.writeInt8 = t9.writeInt8, e12.writeInt16LE = t9.writeInt16LE, e12.writeInt16BE = t9.writeInt16BE, e12.writeInt32LE = t9.writeInt32LE, e12.writeInt32BE = t9.writeInt32BE, e12.writeFloatLE = t9.writeFloatLE, e12.writeFloatBE = t9.writeFloatBE, e12.writeDoubleLE = t9.writeDoubleLE, e12.writeDoubleBE = t9.writeDoubleBE, e12.fill = t9.fill, e12.inspect = t9.inspect, e12.toArrayBuffer = t9.toArrayBuffer, e12;
          };
        }.call(this, O5("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, O5("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/buffer/index.js", "/node_modules/gulp-browserify/node_modules/buffer");
      }, { "base64-js": 2, buffer: 3, ieee754: 10, lYpoI2: 11 }], 4: [function(c7, d6, e10) {
        !function(e11, t9, a7, n9, r10, o9, i7, u7, s6) {
          var a7 = c7("buffer").Buffer, f7 = 4, l7 = new a7(f7);
          l7.fill(0);
          d6.exports = { hash: function(e12, t10, n10, r11) {
            for (var o10 = t10(function(e13, t11) {
              e13.length % f7 != 0 && (n11 = e13.length + (f7 - e13.length % f7), e13 = a7.concat([e13, l7], n11));
              for (var n11, r12 = [], o11 = t11 ? e13.readInt32BE : e13.readInt32LE, i9 = 0; i9 < e13.length; i9 += f7)
                r12.push(o11.call(e13, i9));
              return r12;
            }(e12 = a7.isBuffer(e12) ? e12 : new a7(e12), r11), 8 * e12.length), t10 = r11, i8 = new a7(n10), u8 = t10 ? i8.writeInt32BE : i8.writeInt32LE, s7 = 0; s7 < o10.length; s7++)
              u8.call(i8, o10[s7], 4 * s7, true);
            return i8;
          } };
        }.call(this, c7("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c7("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/helpers.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { buffer: 3, lYpoI2: 11 }], 5: [function(v7, e10, _4) {
        !function(l7, c7, u7, d6, h8, p7, g5, y6, w4) {
          var u7 = v7("buffer").Buffer, e11 = v7("./sha"), t9 = v7("./sha256"), n9 = v7("./rng"), b5 = { sha1: e11, sha256: t9, md5: v7("./md5") }, s6 = 64, a7 = new u7(s6);
          function r10(e12, n10) {
            var r11 = b5[e12 = e12 || "sha1"], o10 = [];
            return r11 || i7("algorithm:", e12, "is not yet supported"), { update: function(e13) {
              return u7.isBuffer(e13) || (e13 = new u7(e13)), o10.push(e13), e13.length, this;
            }, digest: function(e13) {
              var t10 = u7.concat(o10), t10 = n10 ? function(e14, t11, n11) {
                u7.isBuffer(t11) || (t11 = new u7(t11)), u7.isBuffer(n11) || (n11 = new u7(n11)), t11.length > s6 ? t11 = e14(t11) : t11.length < s6 && (t11 = u7.concat([t11, a7], s6));
                for (var r12 = new u7(s6), o11 = new u7(s6), i8 = 0; i8 < s6; i8++)
                  r12[i8] = 54 ^ t11[i8], o11[i8] = 92 ^ t11[i8];
                return n11 = e14(u7.concat([r12, n11])), e14(u7.concat([o11, n11]));
              }(r11, n10, t10) : r11(t10);
              return o10 = null, e13 ? t10.toString(e13) : t10;
            } };
          }
          function i7() {
            var e12 = [].slice.call(arguments).join(" ");
            throw new Error([e12, "we accept pull requests", "http://github.com/dominictarr/crypto-browserify"].join("\n"));
          }
          a7.fill(0), _4.createHash = function(e12) {
            return r10(e12);
          }, _4.createHmac = r10, _4.randomBytes = function(e12, t10) {
            if (!t10 || !t10.call)
              return new u7(n9(e12));
            try {
              t10.call(this, void 0, new u7(n9(e12)));
            } catch (e13) {
              t10(e13);
            }
          };
          var o9, f7 = ["createCredentials", "createCipher", "createCipheriv", "createDecipher", "createDecipheriv", "createSign", "createVerify", "createDiffieHellman", "pbkdf2"], m6 = function(e12) {
            _4[e12] = function() {
              i7("sorry,", e12, "is not implemented yet");
            };
          };
          for (o9 in f7)
            m6(f7[o9]);
        }.call(this, v7("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, v7("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/index.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { "./md5": 6, "./rng": 7, "./sha": 8, "./sha256": 9, buffer: 3, lYpoI2: 11 }], 6: [function(w4, b5, e10) {
        !function(e11, r10, o9, i7, u7, a7, f7, l7, y6) {
          var t9 = w4("./helpers");
          function n9(e12, t10) {
            e12[t10 >> 5] |= 128 << t10 % 32, e12[14 + (t10 + 64 >>> 9 << 4)] = t10;
            for (var n10 = 1732584193, r11 = -271733879, o10 = -1732584194, i8 = 271733878, u8 = 0; u8 < e12.length; u8 += 16) {
              var s7 = n10, a8 = r11, f8 = o10, l8 = i8, n10 = c7(n10, r11, o10, i8, e12[u8 + 0], 7, -680876936), i8 = c7(i8, n10, r11, o10, e12[u8 + 1], 12, -389564586), o10 = c7(o10, i8, n10, r11, e12[u8 + 2], 17, 606105819), r11 = c7(r11, o10, i8, n10, e12[u8 + 3], 22, -1044525330);
              n10 = c7(n10, r11, o10, i8, e12[u8 + 4], 7, -176418897), i8 = c7(i8, n10, r11, o10, e12[u8 + 5], 12, 1200080426), o10 = c7(o10, i8, n10, r11, e12[u8 + 6], 17, -1473231341), r11 = c7(r11, o10, i8, n10, e12[u8 + 7], 22, -45705983), n10 = c7(n10, r11, o10, i8, e12[u8 + 8], 7, 1770035416), i8 = c7(i8, n10, r11, o10, e12[u8 + 9], 12, -1958414417), o10 = c7(o10, i8, n10, r11, e12[u8 + 10], 17, -42063), r11 = c7(r11, o10, i8, n10, e12[u8 + 11], 22, -1990404162), n10 = c7(n10, r11, o10, i8, e12[u8 + 12], 7, 1804603682), i8 = c7(i8, n10, r11, o10, e12[u8 + 13], 12, -40341101), o10 = c7(o10, i8, n10, r11, e12[u8 + 14], 17, -1502002290), n10 = d6(n10, r11 = c7(r11, o10, i8, n10, e12[u8 + 15], 22, 1236535329), o10, i8, e12[u8 + 1], 5, -165796510), i8 = d6(i8, n10, r11, o10, e12[u8 + 6], 9, -1069501632), o10 = d6(o10, i8, n10, r11, e12[u8 + 11], 14, 643717713), r11 = d6(r11, o10, i8, n10, e12[u8 + 0], 20, -373897302), n10 = d6(n10, r11, o10, i8, e12[u8 + 5], 5, -701558691), i8 = d6(i8, n10, r11, o10, e12[u8 + 10], 9, 38016083), o10 = d6(o10, i8, n10, r11, e12[u8 + 15], 14, -660478335), r11 = d6(r11, o10, i8, n10, e12[u8 + 4], 20, -405537848), n10 = d6(n10, r11, o10, i8, e12[u8 + 9], 5, 568446438), i8 = d6(i8, n10, r11, o10, e12[u8 + 14], 9, -1019803690), o10 = d6(o10, i8, n10, r11, e12[u8 + 3], 14, -187363961), r11 = d6(r11, o10, i8, n10, e12[u8 + 8], 20, 1163531501), n10 = d6(n10, r11, o10, i8, e12[u8 + 13], 5, -1444681467), i8 = d6(i8, n10, r11, o10, e12[u8 + 2], 9, -51403784), o10 = d6(o10, i8, n10, r11, e12[u8 + 7], 14, 1735328473), n10 = h8(n10, r11 = d6(r11, o10, i8, n10, e12[u8 + 12], 20, -1926607734), o10, i8, e12[u8 + 5], 4, -378558), i8 = h8(i8, n10, r11, o10, e12[u8 + 8], 11, -2022574463), o10 = h8(o10, i8, n10, r11, e12[u8 + 11], 16, 1839030562), r11 = h8(r11, o10, i8, n10, e12[u8 + 14], 23, -35309556), n10 = h8(n10, r11, o10, i8, e12[u8 + 1], 4, -1530992060), i8 = h8(i8, n10, r11, o10, e12[u8 + 4], 11, 1272893353), o10 = h8(o10, i8, n10, r11, e12[u8 + 7], 16, -155497632), r11 = h8(r11, o10, i8, n10, e12[u8 + 10], 23, -1094730640), n10 = h8(n10, r11, o10, i8, e12[u8 + 13], 4, 681279174), i8 = h8(i8, n10, r11, o10, e12[u8 + 0], 11, -358537222), o10 = h8(o10, i8, n10, r11, e12[u8 + 3], 16, -722521979), r11 = h8(r11, o10, i8, n10, e12[u8 + 6], 23, 76029189), n10 = h8(n10, r11, o10, i8, e12[u8 + 9], 4, -640364487), i8 = h8(i8, n10, r11, o10, e12[u8 + 12], 11, -421815835), o10 = h8(o10, i8, n10, r11, e12[u8 + 15], 16, 530742520), n10 = p7(n10, r11 = h8(r11, o10, i8, n10, e12[u8 + 2], 23, -995338651), o10, i8, e12[u8 + 0], 6, -198630844), i8 = p7(i8, n10, r11, o10, e12[u8 + 7], 10, 1126891415), o10 = p7(o10, i8, n10, r11, e12[u8 + 14], 15, -1416354905), r11 = p7(r11, o10, i8, n10, e12[u8 + 5], 21, -57434055), n10 = p7(n10, r11, o10, i8, e12[u8 + 12], 6, 1700485571), i8 = p7(i8, n10, r11, o10, e12[u8 + 3], 10, -1894986606), o10 = p7(o10, i8, n10, r11, e12[u8 + 10], 15, -1051523), r11 = p7(r11, o10, i8, n10, e12[u8 + 1], 21, -2054922799), n10 = p7(n10, r11, o10, i8, e12[u8 + 8], 6, 1873313359), i8 = p7(i8, n10, r11, o10, e12[u8 + 15], 10, -30611744), o10 = p7(o10, i8, n10, r11, e12[u8 + 6], 15, -1560198380), r11 = p7(r11, o10, i8, n10, e12[u8 + 13], 21, 1309151649), n10 = p7(n10, r11, o10, i8, e12[u8 + 4], 6, -145523070), i8 = p7(i8, n10, r11, o10, e12[u8 + 11], 10, -1120210379), o10 = p7(o10, i8, n10, r11, e12[u8 + 2], 15, 718787259), r11 = p7(r11, o10, i8, n10, e12[u8 + 9], 21, -343485551), n10 = g5(n10, s7), r11 = g5(r11, a8), o10 = g5(o10, f8), i8 = g5(i8, l8);
            }
            return Array(n10, r11, o10, i8);
          }
          function s6(e12, t10, n10, r11, o10, i8) {
            return g5((t10 = g5(g5(t10, e12), g5(r11, i8))) << o10 | t10 >>> 32 - o10, n10);
          }
          function c7(e12, t10, n10, r11, o10, i8, u8) {
            return s6(t10 & n10 | ~t10 & r11, e12, t10, o10, i8, u8);
          }
          function d6(e12, t10, n10, r11, o10, i8, u8) {
            return s6(t10 & r11 | n10 & ~r11, e12, t10, o10, i8, u8);
          }
          function h8(e12, t10, n10, r11, o10, i8, u8) {
            return s6(t10 ^ n10 ^ r11, e12, t10, o10, i8, u8);
          }
          function p7(e12, t10, n10, r11, o10, i8, u8) {
            return s6(n10 ^ (t10 | ~r11), e12, t10, o10, i8, u8);
          }
          function g5(e12, t10) {
            var n10 = (65535 & e12) + (65535 & t10);
            return (e12 >> 16) + (t10 >> 16) + (n10 >> 16) << 16 | 65535 & n10;
          }
          b5.exports = function(e12) {
            return t9.hash(e12, n9, 16);
          };
        }.call(this, w4("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, w4("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/md5.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 7: [function(e10, l7, t9) {
        !function(e11, t10, n9, r10, o9, i7, u7, s6, f7) {
          l7.exports = function(e12) {
            for (var t11, n10 = new Array(e12), r11 = 0; r11 < e12; r11++)
              0 == (3 & r11) && (t11 = 4294967296 * Math.random()), n10[r11] = t11 >>> ((3 & r11) << 3) & 255;
            return n10;
          };
        }.call(this, e10("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e10("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/rng.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { buffer: 3, lYpoI2: 11 }], 8: [function(c7, d6, e10) {
        !function(e11, t9, n9, r10, o9, s6, a7, f7, l7) {
          var i7 = c7("./helpers");
          function u7(l8, c8) {
            l8[c8 >> 5] |= 128 << 24 - c8 % 32, l8[15 + (c8 + 64 >> 9 << 4)] = c8;
            for (var e12, t10, n10, r11 = Array(80), o10 = 1732584193, i8 = -271733879, u8 = -1732584194, s7 = 271733878, d7 = -1009589776, h8 = 0; h8 < l8.length; h8 += 16) {
              for (var p7 = o10, g5 = i8, y6 = u8, w4 = s7, b5 = d7, a8 = 0; a8 < 80; a8++) {
                r11[a8] = a8 < 16 ? l8[h8 + a8] : v7(r11[a8 - 3] ^ r11[a8 - 8] ^ r11[a8 - 14] ^ r11[a8 - 16], 1);
                var f8 = m6(m6(v7(o10, 5), (f8 = i8, t10 = u8, n10 = s7, (e12 = a8) < 20 ? f8 & t10 | ~f8 & n10 : !(e12 < 40) && e12 < 60 ? f8 & t10 | f8 & n10 | t10 & n10 : f8 ^ t10 ^ n10)), m6(m6(d7, r11[a8]), (e12 = a8) < 20 ? 1518500249 : e12 < 40 ? 1859775393 : e12 < 60 ? -1894007588 : -899497514)), d7 = s7, s7 = u8, u8 = v7(i8, 30), i8 = o10, o10 = f8;
              }
              o10 = m6(o10, p7), i8 = m6(i8, g5), u8 = m6(u8, y6), s7 = m6(s7, w4), d7 = m6(d7, b5);
            }
            return Array(o10, i8, u8, s7, d7);
          }
          function m6(e12, t10) {
            var n10 = (65535 & e12) + (65535 & t10);
            return (e12 >> 16) + (t10 >> 16) + (n10 >> 16) << 16 | 65535 & n10;
          }
          function v7(e12, t10) {
            return e12 << t10 | e12 >>> 32 - t10;
          }
          d6.exports = function(e12) {
            return i7.hash(e12, u7, 20, true);
          };
        }.call(this, c7("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c7("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/sha.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 9: [function(c7, d6, e10) {
        !function(e11, t9, n9, r10, u7, s6, a7, f7, l7) {
          function b5(e12, t10) {
            var n10 = (65535 & e12) + (65535 & t10);
            return (e12 >> 16) + (t10 >> 16) + (n10 >> 16) << 16 | 65535 & n10;
          }
          function o9(e12, l8) {
            var c8, d7 = new Array(1116352408, 1899447441, 3049323471, 3921009573, 961987163, 1508970993, 2453635748, 2870763221, 3624381080, 310598401, 607225278, 1426881987, 1925078388, 2162078206, 2614888103, 3248222580, 3835390401, 4022224774, 264347078, 604807628, 770255983, 1249150122, 1555081692, 1996064986, 2554220882, 2821834349, 2952996808, 3210313671, 3336571891, 3584528711, 113926993, 338241895, 666307205, 773529912, 1294757372, 1396182291, 1695183700, 1986661051, 2177026350, 2456956037, 2730485921, 2820302411, 3259730800, 3345764771, 3516065817, 3600352804, 4094571909, 275423344, 430227734, 506948616, 659060556, 883997877, 958139571, 1322822218, 1537002063, 1747873779, 1955562222, 2024104815, 2227730452, 2361852424, 2428436474, 2756734187, 3204031479, 3329325298), t10 = new Array(1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225), n10 = new Array(64);
            e12[l8 >> 5] |= 128 << 24 - l8 % 32, e12[15 + (l8 + 64 >> 9 << 4)] = l8;
            for (var r11, o10, h8 = 0; h8 < e12.length; h8 += 16) {
              for (var i8 = t10[0], u8 = t10[1], s7 = t10[2], p7 = t10[3], a8 = t10[4], g5 = t10[5], y6 = t10[6], w4 = t10[7], f8 = 0; f8 < 64; f8++)
                n10[f8] = f8 < 16 ? e12[f8 + h8] : b5(b5(b5((o10 = n10[f8 - 2], m6(o10, 17) ^ m6(o10, 19) ^ v7(o10, 10)), n10[f8 - 7]), (o10 = n10[f8 - 15], m6(o10, 7) ^ m6(o10, 18) ^ v7(o10, 3))), n10[f8 - 16]), c8 = b5(b5(b5(b5(w4, m6(o10 = a8, 6) ^ m6(o10, 11) ^ m6(o10, 25)), a8 & g5 ^ ~a8 & y6), d7[f8]), n10[f8]), r11 = b5(m6(r11 = i8, 2) ^ m6(r11, 13) ^ m6(r11, 22), i8 & u8 ^ i8 & s7 ^ u8 & s7), w4 = y6, y6 = g5, g5 = a8, a8 = b5(p7, c8), p7 = s7, s7 = u8, u8 = i8, i8 = b5(c8, r11);
              t10[0] = b5(i8, t10[0]), t10[1] = b5(u8, t10[1]), t10[2] = b5(s7, t10[2]), t10[3] = b5(p7, t10[3]), t10[4] = b5(a8, t10[4]), t10[5] = b5(g5, t10[5]), t10[6] = b5(y6, t10[6]), t10[7] = b5(w4, t10[7]);
            }
            return t10;
          }
          var i7 = c7("./helpers"), m6 = function(e12, t10) {
            return e12 >>> t10 | e12 << 32 - t10;
          }, v7 = function(e12, t10) {
            return e12 >>> t10;
          };
          d6.exports = function(e12) {
            return i7.hash(e12, o9, 32, true);
          };
        }.call(this, c7("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c7("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/sha256.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
      }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 10: [function(e10, t9, f7) {
        !function(e11, t10, n9, r10, o9, i7, u7, s6, a7) {
          f7.read = function(e12, t11, n10, r11, o10) {
            var i8, u8, l7 = 8 * o10 - r11 - 1, c7 = (1 << l7) - 1, d6 = c7 >> 1, s7 = -7, a8 = n10 ? o10 - 1 : 0, f8 = n10 ? -1 : 1, o10 = e12[t11 + a8];
            for (a8 += f8, i8 = o10 & (1 << -s7) - 1, o10 >>= -s7, s7 += l7; 0 < s7; i8 = 256 * i8 + e12[t11 + a8], a8 += f8, s7 -= 8)
              ;
            for (u8 = i8 & (1 << -s7) - 1, i8 >>= -s7, s7 += r11; 0 < s7; u8 = 256 * u8 + e12[t11 + a8], a8 += f8, s7 -= 8)
              ;
            if (0 === i8)
              i8 = 1 - d6;
            else {
              if (i8 === c7)
                return u8 ? NaN : 1 / 0 * (o10 ? -1 : 1);
              u8 += Math.pow(2, r11), i8 -= d6;
            }
            return (o10 ? -1 : 1) * u8 * Math.pow(2, i8 - r11);
          }, f7.write = function(e12, t11, l7, n10, r11, c7) {
            var o10, i8, u8 = 8 * c7 - r11 - 1, s7 = (1 << u8) - 1, a8 = s7 >> 1, d6 = 23 === r11 ? Math.pow(2, -24) - Math.pow(2, -77) : 0, f8 = n10 ? 0 : c7 - 1, h8 = n10 ? 1 : -1, c7 = t11 < 0 || 0 === t11 && 1 / t11 < 0 ? 1 : 0;
            for (t11 = Math.abs(t11), isNaN(t11) || t11 === 1 / 0 ? (i8 = isNaN(t11) ? 1 : 0, o10 = s7) : (o10 = Math.floor(Math.log(t11) / Math.LN2), t11 * (n10 = Math.pow(2, -o10)) < 1 && (o10--, n10 *= 2), 2 <= (t11 += 1 <= o10 + a8 ? d6 / n10 : d6 * Math.pow(2, 1 - a8)) * n10 && (o10++, n10 /= 2), s7 <= o10 + a8 ? (i8 = 0, o10 = s7) : 1 <= o10 + a8 ? (i8 = (t11 * n10 - 1) * Math.pow(2, r11), o10 += a8) : (i8 = t11 * Math.pow(2, a8 - 1) * Math.pow(2, r11), o10 = 0)); 8 <= r11; e12[l7 + f8] = 255 & i8, f8 += h8, i8 /= 256, r11 -= 8)
              ;
            for (o10 = o10 << r11 | i8, u8 += r11; 0 < u8; e12[l7 + f8] = 255 & o10, f8 += h8, o10 /= 256, u8 -= 8)
              ;
            e12[l7 + f8 - h8] |= 128 * c7;
          };
        }.call(this, e10("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e10("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/ieee754/index.js", "/node_modules/gulp-browserify/node_modules/ieee754");
      }, { buffer: 3, lYpoI2: 11 }], 11: [function(e10, h8, t9) {
        !function(e11, t10, n9, r10, o9, f7, l7, c7, d6) {
          var i7, u7, s6;
          function a7() {
          }
          (e11 = h8.exports = {}).nextTick = (u7 = "undefined" != typeof window && window.setImmediate, s6 = "undefined" != typeof window && window.postMessage && window.addEventListener, u7 ? function(e12) {
            return window.setImmediate(e12);
          } : s6 ? (i7 = [], window.addEventListener("message", function(e12) {
            var t11 = e12.source;
            t11 !== window && null !== t11 || "process-tick" !== e12.data || (e12.stopPropagation(), 0 < i7.length && i7.shift()());
          }, true), function(e12) {
            i7.push(e12), window.postMessage("process-tick", "*");
          }) : function(e12) {
            setTimeout(e12, 0);
          }), e11.title = "browser", e11.browser = true, e11.env = {}, e11.argv = [], e11.on = a7, e11.addListener = a7, e11.once = a7, e11.off = a7, e11.removeListener = a7, e11.removeAllListeners = a7, e11.emit = a7, e11.binding = function(e12) {
            throw new Error("process.binding is not supported");
          }, e11.cwd = function() {
            return "/";
          }, e11.chdir = function(e12) {
            throw new Error("process.chdir is not supported");
          };
        }.call(this, e10("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e10("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/process/browser.js", "/node_modules/gulp-browserify/node_modules/process");
      }, { buffer: 3, lYpoI2: 11 }] }, {}, [1])(1);
    });
  }
});

// node_modules/object-sizeof/byte_size.js
var require_byte_size = __commonJS({
  "node_modules/object-sizeof/byte_size.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    module.exports = {
      STRING: 2,
      BOOLEAN: 4,
      BYTES: 4,
      NUMBER: 8,
      Int8Array: 1,
      Uint8Array: 1,
      Uint8ClampedArray: 1,
      Int16Array: 2,
      Uint16Array: 2,
      Int32Array: 4,
      Uint32Array: 4,
      Float32Array: 4,
      Float64Array: 8
    };
  }
});

// node_modules/base64-js/index.js
var require_base64_js = __commonJS({
  "node_modules/base64-js/index.js"(exports10) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    exports10.byteLength = byteLength;
    exports10.toByteArray = toByteArray;
    exports10.fromByteArray = fromByteArray;
    var lookup = [];
    var revLookup = [];
    var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
    var code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (i7 = 0, len = code.length; i7 < len; ++i7) {
      lookup[i7] = code[i7];
      revLookup[code.charCodeAt(i7)] = i7;
    }
    var i7;
    var len;
    revLookup["-".charCodeAt(0)] = 62;
    revLookup["_".charCodeAt(0)] = 63;
    function getLens(b64) {
      var len2 = b64.length;
      if (len2 % 4 > 0) {
        throw new Error("Invalid string. Length must be a multiple of 4");
      }
      var validLen = b64.indexOf("=");
      if (validLen === -1)
        validLen = len2;
      var placeHoldersLen = validLen === len2 ? 0 : 4 - validLen % 4;
      return [validLen, placeHoldersLen];
    }
    function byteLength(b64) {
      var lens = getLens(b64);
      var validLen = lens[0];
      var placeHoldersLen = lens[1];
      return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
    }
    function _byteLength(b64, validLen, placeHoldersLen) {
      return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
    }
    function toByteArray(b64) {
      var tmp;
      var lens = getLens(b64);
      var validLen = lens[0];
      var placeHoldersLen = lens[1];
      var arr = new Arr(_byteLength(b64, validLen, placeHoldersLen));
      var curByte = 0;
      var len2 = placeHoldersLen > 0 ? validLen - 4 : validLen;
      var i8;
      for (i8 = 0; i8 < len2; i8 += 4) {
        tmp = revLookup[b64.charCodeAt(i8)] << 18 | revLookup[b64.charCodeAt(i8 + 1)] << 12 | revLookup[b64.charCodeAt(i8 + 2)] << 6 | revLookup[b64.charCodeAt(i8 + 3)];
        arr[curByte++] = tmp >> 16 & 255;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 2) {
        tmp = revLookup[b64.charCodeAt(i8)] << 2 | revLookup[b64.charCodeAt(i8 + 1)] >> 4;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 1) {
        tmp = revLookup[b64.charCodeAt(i8)] << 10 | revLookup[b64.charCodeAt(i8 + 1)] << 4 | revLookup[b64.charCodeAt(i8 + 2)] >> 2;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      return arr;
    }
    function tripletToBase64(num) {
      return lookup[num >> 18 & 63] + lookup[num >> 12 & 63] + lookup[num >> 6 & 63] + lookup[num & 63];
    }
    function encodeChunk(uint8, start, end) {
      var tmp;
      var output = [];
      for (var i8 = start; i8 < end; i8 += 3) {
        tmp = (uint8[i8] << 16 & 16711680) + (uint8[i8 + 1] << 8 & 65280) + (uint8[i8 + 2] & 255);
        output.push(tripletToBase64(tmp));
      }
      return output.join("");
    }
    function fromByteArray(uint8) {
      var tmp;
      var len2 = uint8.length;
      var extraBytes = len2 % 3;
      var parts = [];
      var maxChunkLength = 16383;
      for (var i8 = 0, len22 = len2 - extraBytes; i8 < len22; i8 += maxChunkLength) {
        parts.push(encodeChunk(uint8, i8, i8 + maxChunkLength > len22 ? len22 : i8 + maxChunkLength));
      }
      if (extraBytes === 1) {
        tmp = uint8[len2 - 1];
        parts.push(
          lookup[tmp >> 2] + lookup[tmp << 4 & 63] + "=="
        );
      } else if (extraBytes === 2) {
        tmp = (uint8[len2 - 2] << 8) + uint8[len2 - 1];
        parts.push(
          lookup[tmp >> 10] + lookup[tmp >> 4 & 63] + lookup[tmp << 2 & 63] + "="
        );
      }
      return parts.join("");
    }
  }
});

// node_modules/ieee754/index.js
var require_ieee754 = __commonJS({
  "node_modules/ieee754/index.js"(exports10) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    exports10.read = function(buffer2, offset, isLE, mLen, nBytes) {
      var e10, m6;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var nBits = -7;
      var i7 = isLE ? nBytes - 1 : 0;
      var d6 = isLE ? -1 : 1;
      var s6 = buffer2[offset + i7];
      i7 += d6;
      e10 = s6 & (1 << -nBits) - 1;
      s6 >>= -nBits;
      nBits += eLen;
      for (; nBits > 0; e10 = e10 * 256 + buffer2[offset + i7], i7 += d6, nBits -= 8) {
      }
      m6 = e10 & (1 << -nBits) - 1;
      e10 >>= -nBits;
      nBits += mLen;
      for (; nBits > 0; m6 = m6 * 256 + buffer2[offset + i7], i7 += d6, nBits -= 8) {
      }
      if (e10 === 0) {
        e10 = 1 - eBias;
      } else if (e10 === eMax) {
        return m6 ? NaN : (s6 ? -1 : 1) * Infinity;
      } else {
        m6 = m6 + Math.pow(2, mLen);
        e10 = e10 - eBias;
      }
      return (s6 ? -1 : 1) * m6 * Math.pow(2, e10 - mLen);
    };
    exports10.write = function(buffer2, value, offset, isLE, mLen, nBytes) {
      var e10, m6, c7;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
      var i7 = isLE ? 0 : nBytes - 1;
      var d6 = isLE ? 1 : -1;
      var s6 = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
      value = Math.abs(value);
      if (isNaN(value) || value === Infinity) {
        m6 = isNaN(value) ? 1 : 0;
        e10 = eMax;
      } else {
        e10 = Math.floor(Math.log(value) / Math.LN2);
        if (value * (c7 = Math.pow(2, -e10)) < 1) {
          e10--;
          c7 *= 2;
        }
        if (e10 + eBias >= 1) {
          value += rt / c7;
        } else {
          value += rt * Math.pow(2, 1 - eBias);
        }
        if (value * c7 >= 2) {
          e10++;
          c7 /= 2;
        }
        if (e10 + eBias >= eMax) {
          m6 = 0;
          e10 = eMax;
        } else if (e10 + eBias >= 1) {
          m6 = (value * c7 - 1) * Math.pow(2, mLen);
          e10 = e10 + eBias;
        } else {
          m6 = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
          e10 = 0;
        }
      }
      for (; mLen >= 8; buffer2[offset + i7] = m6 & 255, i7 += d6, m6 /= 256, mLen -= 8) {
      }
      e10 = e10 << mLen | m6;
      eLen += mLen;
      for (; eLen > 0; buffer2[offset + i7] = e10 & 255, i7 += d6, e10 /= 256, eLen -= 8) {
      }
      buffer2[offset + i7 - d6] |= s6 * 128;
    };
  }
});

// node_modules/buffer/index.js
var require_buffer = __commonJS({
  "node_modules/buffer/index.js"(exports10) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var base642 = require_base64_js();
    var ieee754 = require_ieee754();
    var customInspectSymbol = typeof Symbol === "function" && typeof Symbol["for"] === "function" ? Symbol["for"]("nodejs.util.inspect.custom") : null;
    exports10.Buffer = Buffer3;
    exports10.SlowBuffer = SlowBuffer;
    exports10.INSPECT_MAX_BYTES = 50;
    var K_MAX_LENGTH = 2147483647;
    exports10.kMaxLength = K_MAX_LENGTH;
    Buffer3.TYPED_ARRAY_SUPPORT = typedArraySupport();
    if (!Buffer3.TYPED_ARRAY_SUPPORT && typeof console !== "undefined" && typeof console.error === "function") {
      console.error(
        "This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support."
      );
    }
    function typedArraySupport() {
      try {
        const arr = new Uint8Array(1);
        const proto = { foo: function() {
          return 42;
        } };
        Object.setPrototypeOf(proto, Uint8Array.prototype);
        Object.setPrototypeOf(arr, proto);
        return arr.foo() === 42;
      } catch (e10) {
        return false;
      }
    }
    Object.defineProperty(Buffer3.prototype, "parent", {
      enumerable: true,
      get: function() {
        if (!Buffer3.isBuffer(this))
          return void 0;
        return this.buffer;
      }
    });
    Object.defineProperty(Buffer3.prototype, "offset", {
      enumerable: true,
      get: function() {
        if (!Buffer3.isBuffer(this))
          return void 0;
        return this.byteOffset;
      }
    });
    function createBuffer(length) {
      if (length > K_MAX_LENGTH) {
        throw new RangeError('The value "' + length + '" is invalid for option "size"');
      }
      const buf = new Uint8Array(length);
      Object.setPrototypeOf(buf, Buffer3.prototype);
      return buf;
    }
    function Buffer3(arg, encodingOrOffset, length) {
      if (typeof arg === "number") {
        if (typeof encodingOrOffset === "string") {
          throw new TypeError(
            'The "string" argument must be of type string. Received type number'
          );
        }
        return allocUnsafe(arg);
      }
      return from(arg, encodingOrOffset, length);
    }
    Buffer3.poolSize = 8192;
    function from(value, encodingOrOffset, length) {
      if (typeof value === "string") {
        return fromString(value, encodingOrOffset);
      }
      if (ArrayBuffer.isView(value)) {
        return fromArrayView(value);
      }
      if (value == null) {
        throw new TypeError(
          "The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value
        );
      }
      if (isInstance(value, ArrayBuffer) || value && isInstance(value.buffer, ArrayBuffer)) {
        return fromArrayBuffer(value, encodingOrOffset, length);
      }
      if (typeof SharedArrayBuffer !== "undefined" && (isInstance(value, SharedArrayBuffer) || value && isInstance(value.buffer, SharedArrayBuffer))) {
        return fromArrayBuffer(value, encodingOrOffset, length);
      }
      if (typeof value === "number") {
        throw new TypeError(
          'The "value" argument must not be of type number. Received type number'
        );
      }
      const valueOf = value.valueOf && value.valueOf();
      if (valueOf != null && valueOf !== value) {
        return Buffer3.from(valueOf, encodingOrOffset, length);
      }
      const b5 = fromObject(value);
      if (b5)
        return b5;
      if (typeof Symbol !== "undefined" && Symbol.toPrimitive != null && typeof value[Symbol.toPrimitive] === "function") {
        return Buffer3.from(value[Symbol.toPrimitive]("string"), encodingOrOffset, length);
      }
      throw new TypeError(
        "The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value
      );
    }
    Buffer3.from = function(value, encodingOrOffset, length) {
      return from(value, encodingOrOffset, length);
    };
    Object.setPrototypeOf(Buffer3.prototype, Uint8Array.prototype);
    Object.setPrototypeOf(Buffer3, Uint8Array);
    function assertSize(size) {
      if (typeof size !== "number") {
        throw new TypeError('"size" argument must be of type number');
      } else if (size < 0) {
        throw new RangeError('The value "' + size + '" is invalid for option "size"');
      }
    }
    function alloc(size, fill, encoding) {
      assertSize(size);
      if (size <= 0) {
        return createBuffer(size);
      }
      if (fill !== void 0) {
        return typeof encoding === "string" ? createBuffer(size).fill(fill, encoding) : createBuffer(size).fill(fill);
      }
      return createBuffer(size);
    }
    Buffer3.alloc = function(size, fill, encoding) {
      return alloc(size, fill, encoding);
    };
    function allocUnsafe(size) {
      assertSize(size);
      return createBuffer(size < 0 ? 0 : checked(size) | 0);
    }
    Buffer3.allocUnsafe = function(size) {
      return allocUnsafe(size);
    };
    Buffer3.allocUnsafeSlow = function(size) {
      return allocUnsafe(size);
    };
    function fromString(string, encoding) {
      if (typeof encoding !== "string" || encoding === "") {
        encoding = "utf8";
      }
      if (!Buffer3.isEncoding(encoding)) {
        throw new TypeError("Unknown encoding: " + encoding);
      }
      const length = byteLength(string, encoding) | 0;
      let buf = createBuffer(length);
      const actual = buf.write(string, encoding);
      if (actual !== length) {
        buf = buf.slice(0, actual);
      }
      return buf;
    }
    function fromArrayLike(array) {
      const length = array.length < 0 ? 0 : checked(array.length) | 0;
      const buf = createBuffer(length);
      for (let i7 = 0; i7 < length; i7 += 1) {
        buf[i7] = array[i7] & 255;
      }
      return buf;
    }
    function fromArrayView(arrayView) {
      if (isInstance(arrayView, Uint8Array)) {
        const copy = new Uint8Array(arrayView);
        return fromArrayBuffer(copy.buffer, copy.byteOffset, copy.byteLength);
      }
      return fromArrayLike(arrayView);
    }
    function fromArrayBuffer(array, byteOffset, length) {
      if (byteOffset < 0 || array.byteLength < byteOffset) {
        throw new RangeError('"offset" is outside of buffer bounds');
      }
      if (array.byteLength < byteOffset + (length || 0)) {
        throw new RangeError('"length" is outside of buffer bounds');
      }
      let buf;
      if (byteOffset === void 0 && length === void 0) {
        buf = new Uint8Array(array);
      } else if (length === void 0) {
        buf = new Uint8Array(array, byteOffset);
      } else {
        buf = new Uint8Array(array, byteOffset, length);
      }
      Object.setPrototypeOf(buf, Buffer3.prototype);
      return buf;
    }
    function fromObject(obj) {
      if (Buffer3.isBuffer(obj)) {
        const len = checked(obj.length) | 0;
        const buf = createBuffer(len);
        if (buf.length === 0) {
          return buf;
        }
        obj.copy(buf, 0, 0, len);
        return buf;
      }
      if (obj.length !== void 0) {
        if (typeof obj.length !== "number" || numberIsNaN(obj.length)) {
          return createBuffer(0);
        }
        return fromArrayLike(obj);
      }
      if (obj.type === "Buffer" && Array.isArray(obj.data)) {
        return fromArrayLike(obj.data);
      }
    }
    function checked(length) {
      if (length >= K_MAX_LENGTH) {
        throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + K_MAX_LENGTH.toString(16) + " bytes");
      }
      return length | 0;
    }
    function SlowBuffer(length) {
      if (+length != length) {
        length = 0;
      }
      return Buffer3.alloc(+length);
    }
    Buffer3.isBuffer = function isBuffer4(b5) {
      return b5 != null && b5._isBuffer === true && b5 !== Buffer3.prototype;
    };
    Buffer3.compare = function compare(a7, b5) {
      if (isInstance(a7, Uint8Array))
        a7 = Buffer3.from(a7, a7.offset, a7.byteLength);
      if (isInstance(b5, Uint8Array))
        b5 = Buffer3.from(b5, b5.offset, b5.byteLength);
      if (!Buffer3.isBuffer(a7) || !Buffer3.isBuffer(b5)) {
        throw new TypeError(
          'The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array'
        );
      }
      if (a7 === b5)
        return 0;
      let x4 = a7.length;
      let y6 = b5.length;
      for (let i7 = 0, len = Math.min(x4, y6); i7 < len; ++i7) {
        if (a7[i7] !== b5[i7]) {
          x4 = a7[i7];
          y6 = b5[i7];
          break;
        }
      }
      if (x4 < y6)
        return -1;
      if (y6 < x4)
        return 1;
      return 0;
    };
    Buffer3.isEncoding = function isEncoding(encoding) {
      switch (String(encoding).toLowerCase()) {
        case "hex":
        case "utf8":
        case "utf-8":
        case "ascii":
        case "latin1":
        case "binary":
        case "base64":
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return true;
        default:
          return false;
      }
    };
    Buffer3.concat = function concat(list, length) {
      if (!Array.isArray(list)) {
        throw new TypeError('"list" argument must be an Array of Buffers');
      }
      if (list.length === 0) {
        return Buffer3.alloc(0);
      }
      let i7;
      if (length === void 0) {
        length = 0;
        for (i7 = 0; i7 < list.length; ++i7) {
          length += list[i7].length;
        }
      }
      const buffer2 = Buffer3.allocUnsafe(length);
      let pos = 0;
      for (i7 = 0; i7 < list.length; ++i7) {
        let buf = list[i7];
        if (isInstance(buf, Uint8Array)) {
          if (pos + buf.length > buffer2.length) {
            if (!Buffer3.isBuffer(buf))
              buf = Buffer3.from(buf);
            buf.copy(buffer2, pos);
          } else {
            Uint8Array.prototype.set.call(
              buffer2,
              buf,
              pos
            );
          }
        } else if (!Buffer3.isBuffer(buf)) {
          throw new TypeError('"list" argument must be an Array of Buffers');
        } else {
          buf.copy(buffer2, pos);
        }
        pos += buf.length;
      }
      return buffer2;
    };
    function byteLength(string, encoding) {
      if (Buffer3.isBuffer(string)) {
        return string.length;
      }
      if (ArrayBuffer.isView(string) || isInstance(string, ArrayBuffer)) {
        return string.byteLength;
      }
      if (typeof string !== "string") {
        throw new TypeError(
          'The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type ' + typeof string
        );
      }
      const len = string.length;
      const mustMatch = arguments.length > 2 && arguments[2] === true;
      if (!mustMatch && len === 0)
        return 0;
      let loweredCase = false;
      for (; ; ) {
        switch (encoding) {
          case "ascii":
          case "latin1":
          case "binary":
            return len;
          case "utf8":
          case "utf-8":
            return utf8ToBytes(string).length;
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return len * 2;
          case "hex":
            return len >>> 1;
          case "base64":
            return base64ToBytes(string).length;
          default:
            if (loweredCase) {
              return mustMatch ? -1 : utf8ToBytes(string).length;
            }
            encoding = ("" + encoding).toLowerCase();
            loweredCase = true;
        }
      }
    }
    Buffer3.byteLength = byteLength;
    function slowToString(encoding, start, end) {
      let loweredCase = false;
      if (start === void 0 || start < 0) {
        start = 0;
      }
      if (start > this.length) {
        return "";
      }
      if (end === void 0 || end > this.length) {
        end = this.length;
      }
      if (end <= 0) {
        return "";
      }
      end >>>= 0;
      start >>>= 0;
      if (end <= start) {
        return "";
      }
      if (!encoding)
        encoding = "utf8";
      while (true) {
        switch (encoding) {
          case "hex":
            return hexSlice(this, start, end);
          case "utf8":
          case "utf-8":
            return utf8Slice(this, start, end);
          case "ascii":
            return asciiSlice(this, start, end);
          case "latin1":
          case "binary":
            return latin1Slice(this, start, end);
          case "base64":
            return base64Slice(this, start, end);
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return utf16leSlice(this, start, end);
          default:
            if (loweredCase)
              throw new TypeError("Unknown encoding: " + encoding);
            encoding = (encoding + "").toLowerCase();
            loweredCase = true;
        }
      }
    }
    Buffer3.prototype._isBuffer = true;
    function swap(b5, n9, m6) {
      const i7 = b5[n9];
      b5[n9] = b5[m6];
      b5[m6] = i7;
    }
    Buffer3.prototype.swap16 = function swap16() {
      const len = this.length;
      if (len % 2 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 16-bits");
      }
      for (let i7 = 0; i7 < len; i7 += 2) {
        swap(this, i7, i7 + 1);
      }
      return this;
    };
    Buffer3.prototype.swap32 = function swap32() {
      const len = this.length;
      if (len % 4 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 32-bits");
      }
      for (let i7 = 0; i7 < len; i7 += 4) {
        swap(this, i7, i7 + 3);
        swap(this, i7 + 1, i7 + 2);
      }
      return this;
    };
    Buffer3.prototype.swap64 = function swap64() {
      const len = this.length;
      if (len % 8 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 64-bits");
      }
      for (let i7 = 0; i7 < len; i7 += 8) {
        swap(this, i7, i7 + 7);
        swap(this, i7 + 1, i7 + 6);
        swap(this, i7 + 2, i7 + 5);
        swap(this, i7 + 3, i7 + 4);
      }
      return this;
    };
    Buffer3.prototype.toString = function toString3() {
      const length = this.length;
      if (length === 0)
        return "";
      if (arguments.length === 0)
        return utf8Slice(this, 0, length);
      return slowToString.apply(this, arguments);
    };
    Buffer3.prototype.toLocaleString = Buffer3.prototype.toString;
    Buffer3.prototype.equals = function equals(b5) {
      if (!Buffer3.isBuffer(b5))
        throw new TypeError("Argument must be a Buffer");
      if (this === b5)
        return true;
      return Buffer3.compare(this, b5) === 0;
    };
    Buffer3.prototype.inspect = function inspect3() {
      let str = "";
      const max = exports10.INSPECT_MAX_BYTES;
      str = this.toString("hex", 0, max).replace(/(.{2})/g, "$1 ").trim();
      if (this.length > max)
        str += " ... ";
      return "<Buffer " + str + ">";
    };
    if (customInspectSymbol) {
      Buffer3.prototype[customInspectSymbol] = Buffer3.prototype.inspect;
    }
    Buffer3.prototype.compare = function compare(target, start, end, thisStart, thisEnd) {
      if (isInstance(target, Uint8Array)) {
        target = Buffer3.from(target, target.offset, target.byteLength);
      }
      if (!Buffer3.isBuffer(target)) {
        throw new TypeError(
          'The "target" argument must be one of type Buffer or Uint8Array. Received type ' + typeof target
        );
      }
      if (start === void 0) {
        start = 0;
      }
      if (end === void 0) {
        end = target ? target.length : 0;
      }
      if (thisStart === void 0) {
        thisStart = 0;
      }
      if (thisEnd === void 0) {
        thisEnd = this.length;
      }
      if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
        throw new RangeError("out of range index");
      }
      if (thisStart >= thisEnd && start >= end) {
        return 0;
      }
      if (thisStart >= thisEnd) {
        return -1;
      }
      if (start >= end) {
        return 1;
      }
      start >>>= 0;
      end >>>= 0;
      thisStart >>>= 0;
      thisEnd >>>= 0;
      if (this === target)
        return 0;
      let x4 = thisEnd - thisStart;
      let y6 = end - start;
      const len = Math.min(x4, y6);
      const thisCopy = this.slice(thisStart, thisEnd);
      const targetCopy = target.slice(start, end);
      for (let i7 = 0; i7 < len; ++i7) {
        if (thisCopy[i7] !== targetCopy[i7]) {
          x4 = thisCopy[i7];
          y6 = targetCopy[i7];
          break;
        }
      }
      if (x4 < y6)
        return -1;
      if (y6 < x4)
        return 1;
      return 0;
    };
    function bidirectionalIndexOf(buffer2, val, byteOffset, encoding, dir) {
      if (buffer2.length === 0)
        return -1;
      if (typeof byteOffset === "string") {
        encoding = byteOffset;
        byteOffset = 0;
      } else if (byteOffset > 2147483647) {
        byteOffset = 2147483647;
      } else if (byteOffset < -2147483648) {
        byteOffset = -2147483648;
      }
      byteOffset = +byteOffset;
      if (numberIsNaN(byteOffset)) {
        byteOffset = dir ? 0 : buffer2.length - 1;
      }
      if (byteOffset < 0)
        byteOffset = buffer2.length + byteOffset;
      if (byteOffset >= buffer2.length) {
        if (dir)
          return -1;
        else
          byteOffset = buffer2.length - 1;
      } else if (byteOffset < 0) {
        if (dir)
          byteOffset = 0;
        else
          return -1;
      }
      if (typeof val === "string") {
        val = Buffer3.from(val, encoding);
      }
      if (Buffer3.isBuffer(val)) {
        if (val.length === 0) {
          return -1;
        }
        return arrayIndexOf(buffer2, val, byteOffset, encoding, dir);
      } else if (typeof val === "number") {
        val = val & 255;
        if (typeof Uint8Array.prototype.indexOf === "function") {
          if (dir) {
            return Uint8Array.prototype.indexOf.call(buffer2, val, byteOffset);
          } else {
            return Uint8Array.prototype.lastIndexOf.call(buffer2, val, byteOffset);
          }
        }
        return arrayIndexOf(buffer2, [val], byteOffset, encoding, dir);
      }
      throw new TypeError("val must be string, number or Buffer");
    }
    function arrayIndexOf(arr, val, byteOffset, encoding, dir) {
      let indexSize = 1;
      let arrLength = arr.length;
      let valLength = val.length;
      if (encoding !== void 0) {
        encoding = String(encoding).toLowerCase();
        if (encoding === "ucs2" || encoding === "ucs-2" || encoding === "utf16le" || encoding === "utf-16le") {
          if (arr.length < 2 || val.length < 2) {
            return -1;
          }
          indexSize = 2;
          arrLength /= 2;
          valLength /= 2;
          byteOffset /= 2;
        }
      }
      function read2(buf, i8) {
        if (indexSize === 1) {
          return buf[i8];
        } else {
          return buf.readUInt16BE(i8 * indexSize);
        }
      }
      let i7;
      if (dir) {
        let foundIndex = -1;
        for (i7 = byteOffset; i7 < arrLength; i7++) {
          if (read2(arr, i7) === read2(val, foundIndex === -1 ? 0 : i7 - foundIndex)) {
            if (foundIndex === -1)
              foundIndex = i7;
            if (i7 - foundIndex + 1 === valLength)
              return foundIndex * indexSize;
          } else {
            if (foundIndex !== -1)
              i7 -= i7 - foundIndex;
            foundIndex = -1;
          }
        }
      } else {
        if (byteOffset + valLength > arrLength)
          byteOffset = arrLength - valLength;
        for (i7 = byteOffset; i7 >= 0; i7--) {
          let found = true;
          for (let j4 = 0; j4 < valLength; j4++) {
            if (read2(arr, i7 + j4) !== read2(val, j4)) {
              found = false;
              break;
            }
          }
          if (found)
            return i7;
        }
      }
      return -1;
    }
    Buffer3.prototype.includes = function includes(val, byteOffset, encoding) {
      return this.indexOf(val, byteOffset, encoding) !== -1;
    };
    Buffer3.prototype.indexOf = function indexOf(val, byteOffset, encoding) {
      return bidirectionalIndexOf(this, val, byteOffset, encoding, true);
    };
    Buffer3.prototype.lastIndexOf = function lastIndexOf(val, byteOffset, encoding) {
      return bidirectionalIndexOf(this, val, byteOffset, encoding, false);
    };
    function hexWrite(buf, string, offset, length) {
      offset = Number(offset) || 0;
      const remaining = buf.length - offset;
      if (!length) {
        length = remaining;
      } else {
        length = Number(length);
        if (length > remaining) {
          length = remaining;
        }
      }
      const strLen = string.length;
      if (length > strLen / 2) {
        length = strLen / 2;
      }
      let i7;
      for (i7 = 0; i7 < length; ++i7) {
        const parsed = parseInt(string.substr(i7 * 2, 2), 16);
        if (numberIsNaN(parsed))
          return i7;
        buf[offset + i7] = parsed;
      }
      return i7;
    }
    function utf8Write(buf, string, offset, length) {
      return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length);
    }
    function asciiWrite(buf, string, offset, length) {
      return blitBuffer(asciiToBytes(string), buf, offset, length);
    }
    function base64Write(buf, string, offset, length) {
      return blitBuffer(base64ToBytes(string), buf, offset, length);
    }
    function ucs2Write(buf, string, offset, length) {
      return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length);
    }
    Buffer3.prototype.write = function write2(string, offset, length, encoding) {
      if (offset === void 0) {
        encoding = "utf8";
        length = this.length;
        offset = 0;
      } else if (length === void 0 && typeof offset === "string") {
        encoding = offset;
        length = this.length;
        offset = 0;
      } else if (isFinite(offset)) {
        offset = offset >>> 0;
        if (isFinite(length)) {
          length = length >>> 0;
          if (encoding === void 0)
            encoding = "utf8";
        } else {
          encoding = length;
          length = void 0;
        }
      } else {
        throw new Error(
          "Buffer.write(string, encoding, offset[, length]) is no longer supported"
        );
      }
      const remaining = this.length - offset;
      if (length === void 0 || length > remaining)
        length = remaining;
      if (string.length > 0 && (length < 0 || offset < 0) || offset > this.length) {
        throw new RangeError("Attempt to write outside buffer bounds");
      }
      if (!encoding)
        encoding = "utf8";
      let loweredCase = false;
      for (; ; ) {
        switch (encoding) {
          case "hex":
            return hexWrite(this, string, offset, length);
          case "utf8":
          case "utf-8":
            return utf8Write(this, string, offset, length);
          case "ascii":
          case "latin1":
          case "binary":
            return asciiWrite(this, string, offset, length);
          case "base64":
            return base64Write(this, string, offset, length);
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return ucs2Write(this, string, offset, length);
          default:
            if (loweredCase)
              throw new TypeError("Unknown encoding: " + encoding);
            encoding = ("" + encoding).toLowerCase();
            loweredCase = true;
        }
      }
    };
    Buffer3.prototype.toJSON = function toJSON2() {
      return {
        type: "Buffer",
        data: Array.prototype.slice.call(this._arr || this, 0)
      };
    };
    function base64Slice(buf, start, end) {
      if (start === 0 && end === buf.length) {
        return base642.fromByteArray(buf);
      } else {
        return base642.fromByteArray(buf.slice(start, end));
      }
    }
    function utf8Slice(buf, start, end) {
      end = Math.min(buf.length, end);
      const res = [];
      let i7 = start;
      while (i7 < end) {
        const firstByte = buf[i7];
        let codePoint = null;
        let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
        if (i7 + bytesPerSequence <= end) {
          let secondByte, thirdByte, fourthByte, tempCodePoint;
          switch (bytesPerSequence) {
            case 1:
              if (firstByte < 128) {
                codePoint = firstByte;
              }
              break;
            case 2:
              secondByte = buf[i7 + 1];
              if ((secondByte & 192) === 128) {
                tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
                if (tempCodePoint > 127) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 3:
              secondByte = buf[i7 + 1];
              thirdByte = buf[i7 + 2];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
                if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 4:
              secondByte = buf[i7 + 1];
              thirdByte = buf[i7 + 2];
              fourthByte = buf[i7 + 3];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128 && (fourthByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 18 | (secondByte & 63) << 12 | (thirdByte & 63) << 6 | fourthByte & 63;
                if (tempCodePoint > 65535 && tempCodePoint < 1114112) {
                  codePoint = tempCodePoint;
                }
              }
          }
        }
        if (codePoint === null) {
          codePoint = 65533;
          bytesPerSequence = 1;
        } else if (codePoint > 65535) {
          codePoint -= 65536;
          res.push(codePoint >>> 10 & 1023 | 55296);
          codePoint = 56320 | codePoint & 1023;
        }
        res.push(codePoint);
        i7 += bytesPerSequence;
      }
      return decodeCodePointsArray(res);
    }
    var MAX_ARGUMENTS_LENGTH = 4096;
    function decodeCodePointsArray(codePoints) {
      const len = codePoints.length;
      if (len <= MAX_ARGUMENTS_LENGTH) {
        return String.fromCharCode.apply(String, codePoints);
      }
      let res = "";
      let i7 = 0;
      while (i7 < len) {
        res += String.fromCharCode.apply(
          String,
          codePoints.slice(i7, i7 += MAX_ARGUMENTS_LENGTH)
        );
      }
      return res;
    }
    function asciiSlice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i7 = start; i7 < end; ++i7) {
        ret += String.fromCharCode(buf[i7] & 127);
      }
      return ret;
    }
    function latin1Slice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i7 = start; i7 < end; ++i7) {
        ret += String.fromCharCode(buf[i7]);
      }
      return ret;
    }
    function hexSlice(buf, start, end) {
      const len = buf.length;
      if (!start || start < 0)
        start = 0;
      if (!end || end < 0 || end > len)
        end = len;
      let out = "";
      for (let i7 = start; i7 < end; ++i7) {
        out += hexSliceLookupTable[buf[i7]];
      }
      return out;
    }
    function utf16leSlice(buf, start, end) {
      const bytes = buf.slice(start, end);
      let res = "";
      for (let i7 = 0; i7 < bytes.length - 1; i7 += 2) {
        res += String.fromCharCode(bytes[i7] + bytes[i7 + 1] * 256);
      }
      return res;
    }
    Buffer3.prototype.slice = function slice(start, end) {
      const len = this.length;
      start = ~~start;
      end = end === void 0 ? len : ~~end;
      if (start < 0) {
        start += len;
        if (start < 0)
          start = 0;
      } else if (start > len) {
        start = len;
      }
      if (end < 0) {
        end += len;
        if (end < 0)
          end = 0;
      } else if (end > len) {
        end = len;
      }
      if (end < start)
        end = start;
      const newBuf = this.subarray(start, end);
      Object.setPrototypeOf(newBuf, Buffer3.prototype);
      return newBuf;
    };
    function checkOffset(offset, ext, length) {
      if (offset % 1 !== 0 || offset < 0)
        throw new RangeError("offset is not uint");
      if (offset + ext > length)
        throw new RangeError("Trying to access beyond buffer length");
    }
    Buffer3.prototype.readUintLE = Buffer3.prototype.readUIntLE = function readUIntLE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert)
        checkOffset(offset, byteLength2, this.length);
      let val = this[offset];
      let mul = 1;
      let i7 = 0;
      while (++i7 < byteLength2 && (mul *= 256)) {
        val += this[offset + i7] * mul;
      }
      return val;
    };
    Buffer3.prototype.readUintBE = Buffer3.prototype.readUIntBE = function readUIntBE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        checkOffset(offset, byteLength2, this.length);
      }
      let val = this[offset + --byteLength2];
      let mul = 1;
      while (byteLength2 > 0 && (mul *= 256)) {
        val += this[offset + --byteLength2] * mul;
      }
      return val;
    };
    Buffer3.prototype.readUint8 = Buffer3.prototype.readUInt8 = function readUInt8(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 1, this.length);
      return this[offset];
    };
    Buffer3.prototype.readUint16LE = Buffer3.prototype.readUInt16LE = function readUInt16LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 2, this.length);
      return this[offset] | this[offset + 1] << 8;
    };
    Buffer3.prototype.readUint16BE = Buffer3.prototype.readUInt16BE = function readUInt16BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 2, this.length);
      return this[offset] << 8 | this[offset + 1];
    };
    Buffer3.prototype.readUint32LE = Buffer3.prototype.readUInt32LE = function readUInt32LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return (this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16) + this[offset + 3] * 16777216;
    };
    Buffer3.prototype.readUint32BE = Buffer3.prototype.readUInt32BE = function readUInt32BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return this[offset] * 16777216 + (this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3]);
    };
    Buffer3.prototype.readBigUInt64LE = defineBigIntMethod(function readBigUInt64LE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const lo = first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24;
      const hi = this[++offset] + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + last * 2 ** 24;
      return BigInt(lo) + (BigInt(hi) << BigInt(32));
    });
    Buffer3.prototype.readBigUInt64BE = defineBigIntMethod(function readBigUInt64BE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const hi = first * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
      const lo = this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last;
      return (BigInt(hi) << BigInt(32)) + BigInt(lo);
    });
    Buffer3.prototype.readIntLE = function readIntLE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert)
        checkOffset(offset, byteLength2, this.length);
      let val = this[offset];
      let mul = 1;
      let i7 = 0;
      while (++i7 < byteLength2 && (mul *= 256)) {
        val += this[offset + i7] * mul;
      }
      mul *= 128;
      if (val >= mul)
        val -= Math.pow(2, 8 * byteLength2);
      return val;
    };
    Buffer3.prototype.readIntBE = function readIntBE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert)
        checkOffset(offset, byteLength2, this.length);
      let i7 = byteLength2;
      let mul = 1;
      let val = this[offset + --i7];
      while (i7 > 0 && (mul *= 256)) {
        val += this[offset + --i7] * mul;
      }
      mul *= 128;
      if (val >= mul)
        val -= Math.pow(2, 8 * byteLength2);
      return val;
    };
    Buffer3.prototype.readInt8 = function readInt8(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 1, this.length);
      if (!(this[offset] & 128))
        return this[offset];
      return (255 - this[offset] + 1) * -1;
    };
    Buffer3.prototype.readInt16LE = function readInt16LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 2, this.length);
      const val = this[offset] | this[offset + 1] << 8;
      return val & 32768 ? val | 4294901760 : val;
    };
    Buffer3.prototype.readInt16BE = function readInt16BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 2, this.length);
      const val = this[offset + 1] | this[offset] << 8;
      return val & 32768 ? val | 4294901760 : val;
    };
    Buffer3.prototype.readInt32LE = function readInt32LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16 | this[offset + 3] << 24;
    };
    Buffer3.prototype.readInt32BE = function readInt32BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return this[offset] << 24 | this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3];
    };
    Buffer3.prototype.readBigInt64LE = defineBigIntMethod(function readBigInt64LE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const val = this[offset + 4] + this[offset + 5] * 2 ** 8 + this[offset + 6] * 2 ** 16 + (last << 24);
      return (BigInt(val) << BigInt(32)) + BigInt(first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24);
    });
    Buffer3.prototype.readBigInt64BE = defineBigIntMethod(function readBigInt64BE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const val = (first << 24) + // Overflow
      this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
      return (BigInt(val) << BigInt(32)) + BigInt(this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last);
    });
    Buffer3.prototype.readFloatLE = function readFloatLE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return ieee754.read(this, offset, true, 23, 4);
    };
    Buffer3.prototype.readFloatBE = function readFloatBE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 4, this.length);
      return ieee754.read(this, offset, false, 23, 4);
    };
    Buffer3.prototype.readDoubleLE = function readDoubleLE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 8, this.length);
      return ieee754.read(this, offset, true, 52, 8);
    };
    Buffer3.prototype.readDoubleBE = function readDoubleBE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert)
        checkOffset(offset, 8, this.length);
      return ieee754.read(this, offset, false, 52, 8);
    };
    function checkInt(buf, value, offset, ext, max, min) {
      if (!Buffer3.isBuffer(buf))
        throw new TypeError('"buffer" argument must be a Buffer instance');
      if (value > max || value < min)
        throw new RangeError('"value" argument is out of bounds');
      if (offset + ext > buf.length)
        throw new RangeError("Index out of range");
    }
    Buffer3.prototype.writeUintLE = Buffer3.prototype.writeUIntLE = function writeUIntLE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
        checkInt(this, value, offset, byteLength2, maxBytes, 0);
      }
      let mul = 1;
      let i7 = 0;
      this[offset] = value & 255;
      while (++i7 < byteLength2 && (mul *= 256)) {
        this[offset + i7] = value / mul & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeUintBE = Buffer3.prototype.writeUIntBE = function writeUIntBE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
        checkInt(this, value, offset, byteLength2, maxBytes, 0);
      }
      let i7 = byteLength2 - 1;
      let mul = 1;
      this[offset + i7] = value & 255;
      while (--i7 >= 0 && (mul *= 256)) {
        this[offset + i7] = value / mul & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeUint8 = Buffer3.prototype.writeUInt8 = function writeUInt8(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 1, 255, 0);
      this[offset] = value & 255;
      return offset + 1;
    };
    Buffer3.prototype.writeUint16LE = Buffer3.prototype.writeUInt16LE = function writeUInt16LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 2, 65535, 0);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      return offset + 2;
    };
    Buffer3.prototype.writeUint16BE = Buffer3.prototype.writeUInt16BE = function writeUInt16BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 2, 65535, 0);
      this[offset] = value >>> 8;
      this[offset + 1] = value & 255;
      return offset + 2;
    };
    Buffer3.prototype.writeUint32LE = Buffer3.prototype.writeUInt32LE = function writeUInt32LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 4, 4294967295, 0);
      this[offset + 3] = value >>> 24;
      this[offset + 2] = value >>> 16;
      this[offset + 1] = value >>> 8;
      this[offset] = value & 255;
      return offset + 4;
    };
    Buffer3.prototype.writeUint32BE = Buffer3.prototype.writeUInt32BE = function writeUInt32BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 4, 4294967295, 0);
      this[offset] = value >>> 24;
      this[offset + 1] = value >>> 16;
      this[offset + 2] = value >>> 8;
      this[offset + 3] = value & 255;
      return offset + 4;
    };
    function wrtBigUInt64LE(buf, value, offset, min, max) {
      checkIntBI(value, min, max, buf, offset, 7);
      let lo = Number(value & BigInt(4294967295));
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      let hi = Number(value >> BigInt(32) & BigInt(4294967295));
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      return offset;
    }
    function wrtBigUInt64BE(buf, value, offset, min, max) {
      checkIntBI(value, min, max, buf, offset, 7);
      let lo = Number(value & BigInt(4294967295));
      buf[offset + 7] = lo;
      lo = lo >> 8;
      buf[offset + 6] = lo;
      lo = lo >> 8;
      buf[offset + 5] = lo;
      lo = lo >> 8;
      buf[offset + 4] = lo;
      let hi = Number(value >> BigInt(32) & BigInt(4294967295));
      buf[offset + 3] = hi;
      hi = hi >> 8;
      buf[offset + 2] = hi;
      hi = hi >> 8;
      buf[offset + 1] = hi;
      hi = hi >> 8;
      buf[offset] = hi;
      return offset + 8;
    }
    Buffer3.prototype.writeBigUInt64LE = defineBigIntMethod(function writeBigUInt64LE(value, offset = 0) {
      return wrtBigUInt64LE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
    });
    Buffer3.prototype.writeBigUInt64BE = defineBigIntMethod(function writeBigUInt64BE(value, offset = 0) {
      return wrtBigUInt64BE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
    });
    Buffer3.prototype.writeIntLE = function writeIntLE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        const limit = Math.pow(2, 8 * byteLength2 - 1);
        checkInt(this, value, offset, byteLength2, limit - 1, -limit);
      }
      let i7 = 0;
      let mul = 1;
      let sub = 0;
      this[offset] = value & 255;
      while (++i7 < byteLength2 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i7 - 1] !== 0) {
          sub = 1;
        }
        this[offset + i7] = (value / mul >> 0) - sub & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeIntBE = function writeIntBE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        const limit = Math.pow(2, 8 * byteLength2 - 1);
        checkInt(this, value, offset, byteLength2, limit - 1, -limit);
      }
      let i7 = byteLength2 - 1;
      let mul = 1;
      let sub = 0;
      this[offset + i7] = value & 255;
      while (--i7 >= 0 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i7 + 1] !== 0) {
          sub = 1;
        }
        this[offset + i7] = (value / mul >> 0) - sub & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeInt8 = function writeInt8(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 1, 127, -128);
      if (value < 0)
        value = 255 + value + 1;
      this[offset] = value & 255;
      return offset + 1;
    };
    Buffer3.prototype.writeInt16LE = function writeInt16LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 2, 32767, -32768);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      return offset + 2;
    };
    Buffer3.prototype.writeInt16BE = function writeInt16BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 2, 32767, -32768);
      this[offset] = value >>> 8;
      this[offset + 1] = value & 255;
      return offset + 2;
    };
    Buffer3.prototype.writeInt32LE = function writeInt32LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 4, 2147483647, -2147483648);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      this[offset + 2] = value >>> 16;
      this[offset + 3] = value >>> 24;
      return offset + 4;
    };
    Buffer3.prototype.writeInt32BE = function writeInt32BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert)
        checkInt(this, value, offset, 4, 2147483647, -2147483648);
      if (value < 0)
        value = 4294967295 + value + 1;
      this[offset] = value >>> 24;
      this[offset + 1] = value >>> 16;
      this[offset + 2] = value >>> 8;
      this[offset + 3] = value & 255;
      return offset + 4;
    };
    Buffer3.prototype.writeBigInt64LE = defineBigIntMethod(function writeBigInt64LE(value, offset = 0) {
      return wrtBigUInt64LE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
    });
    Buffer3.prototype.writeBigInt64BE = defineBigIntMethod(function writeBigInt64BE(value, offset = 0) {
      return wrtBigUInt64BE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
    });
    function checkIEEE754(buf, value, offset, ext, max, min) {
      if (offset + ext > buf.length)
        throw new RangeError("Index out of range");
      if (offset < 0)
        throw new RangeError("Index out of range");
    }
    function writeFloat(buf, value, offset, littleEndian, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        checkIEEE754(buf, value, offset, 4);
      }
      ieee754.write(buf, value, offset, littleEndian, 23, 4);
      return offset + 4;
    }
    Buffer3.prototype.writeFloatLE = function writeFloatLE(value, offset, noAssert) {
      return writeFloat(this, value, offset, true, noAssert);
    };
    Buffer3.prototype.writeFloatBE = function writeFloatBE(value, offset, noAssert) {
      return writeFloat(this, value, offset, false, noAssert);
    };
    function writeDouble(buf, value, offset, littleEndian, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        checkIEEE754(buf, value, offset, 8);
      }
      ieee754.write(buf, value, offset, littleEndian, 52, 8);
      return offset + 8;
    }
    Buffer3.prototype.writeDoubleLE = function writeDoubleLE(value, offset, noAssert) {
      return writeDouble(this, value, offset, true, noAssert);
    };
    Buffer3.prototype.writeDoubleBE = function writeDoubleBE(value, offset, noAssert) {
      return writeDouble(this, value, offset, false, noAssert);
    };
    Buffer3.prototype.copy = function copy(target, targetStart, start, end) {
      if (!Buffer3.isBuffer(target))
        throw new TypeError("argument should be a Buffer");
      if (!start)
        start = 0;
      if (!end && end !== 0)
        end = this.length;
      if (targetStart >= target.length)
        targetStart = target.length;
      if (!targetStart)
        targetStart = 0;
      if (end > 0 && end < start)
        end = start;
      if (end === start)
        return 0;
      if (target.length === 0 || this.length === 0)
        return 0;
      if (targetStart < 0) {
        throw new RangeError("targetStart out of bounds");
      }
      if (start < 0 || start >= this.length)
        throw new RangeError("Index out of range");
      if (end < 0)
        throw new RangeError("sourceEnd out of bounds");
      if (end > this.length)
        end = this.length;
      if (target.length - targetStart < end - start) {
        end = target.length - targetStart + start;
      }
      const len = end - start;
      if (this === target && typeof Uint8Array.prototype.copyWithin === "function") {
        this.copyWithin(targetStart, start, end);
      } else {
        Uint8Array.prototype.set.call(
          target,
          this.subarray(start, end),
          targetStart
        );
      }
      return len;
    };
    Buffer3.prototype.fill = function fill(val, start, end, encoding) {
      if (typeof val === "string") {
        if (typeof start === "string") {
          encoding = start;
          start = 0;
          end = this.length;
        } else if (typeof end === "string") {
          encoding = end;
          end = this.length;
        }
        if (encoding !== void 0 && typeof encoding !== "string") {
          throw new TypeError("encoding must be a string");
        }
        if (typeof encoding === "string" && !Buffer3.isEncoding(encoding)) {
          throw new TypeError("Unknown encoding: " + encoding);
        }
        if (val.length === 1) {
          const code = val.charCodeAt(0);
          if (encoding === "utf8" && code < 128 || encoding === "latin1") {
            val = code;
          }
        }
      } else if (typeof val === "number") {
        val = val & 255;
      } else if (typeof val === "boolean") {
        val = Number(val);
      }
      if (start < 0 || this.length < start || this.length < end) {
        throw new RangeError("Out of range index");
      }
      if (end <= start) {
        return this;
      }
      start = start >>> 0;
      end = end === void 0 ? this.length : end >>> 0;
      if (!val)
        val = 0;
      let i7;
      if (typeof val === "number") {
        for (i7 = start; i7 < end; ++i7) {
          this[i7] = val;
        }
      } else {
        const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
        const len = bytes.length;
        if (len === 0) {
          throw new TypeError('The value "' + val + '" is invalid for argument "value"');
        }
        for (i7 = 0; i7 < end - start; ++i7) {
          this[i7 + start] = bytes[i7 % len];
        }
      }
      return this;
    };
    var errors = {};
    function E4(sym, getMessage, Base) {
      errors[sym] = class NodeError extends Base {
        constructor() {
          super();
          Object.defineProperty(this, "message", {
            value: getMessage.apply(this, arguments),
            writable: true,
            configurable: true
          });
          this.name = `${this.name} [${sym}]`;
          this.stack;
          delete this.name;
        }
        get code() {
          return sym;
        }
        set code(value) {
          Object.defineProperty(this, "code", {
            configurable: true,
            enumerable: true,
            value,
            writable: true
          });
        }
        toString() {
          return `${this.name} [${sym}]: ${this.message}`;
        }
      };
    }
    E4(
      "ERR_BUFFER_OUT_OF_BOUNDS",
      function(name3) {
        if (name3) {
          return `${name3} is outside of buffer bounds`;
        }
        return "Attempt to access memory outside buffer bounds";
      },
      RangeError
    );
    E4(
      "ERR_INVALID_ARG_TYPE",
      function(name3, actual) {
        return `The "${name3}" argument must be of type number. Received type ${typeof actual}`;
      },
      TypeError
    );
    E4(
      "ERR_OUT_OF_RANGE",
      function(str, range, input) {
        let msg = `The value of "${str}" is out of range.`;
        let received = input;
        if (Number.isInteger(input) && Math.abs(input) > 2 ** 32) {
          received = addNumericalSeparator(String(input));
        } else if (typeof input === "bigint") {
          received = String(input);
          if (input > BigInt(2) ** BigInt(32) || input < -(BigInt(2) ** BigInt(32))) {
            received = addNumericalSeparator(received);
          }
          received += "n";
        }
        msg += ` It must be ${range}. Received ${received}`;
        return msg;
      },
      RangeError
    );
    function addNumericalSeparator(val) {
      let res = "";
      let i7 = val.length;
      const start = val[0] === "-" ? 1 : 0;
      for (; i7 >= start + 4; i7 -= 3) {
        res = `_${val.slice(i7 - 3, i7)}${res}`;
      }
      return `${val.slice(0, i7)}${res}`;
    }
    function checkBounds(buf, offset, byteLength2) {
      validateNumber(offset, "offset");
      if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
        boundsError(offset, buf.length - (byteLength2 + 1));
      }
    }
    function checkIntBI(value, min, max, buf, offset, byteLength2) {
      if (value > max || value < min) {
        const n9 = typeof min === "bigint" ? "n" : "";
        let range;
        if (byteLength2 > 3) {
          if (min === 0 || min === BigInt(0)) {
            range = `>= 0${n9} and < 2${n9} ** ${(byteLength2 + 1) * 8}${n9}`;
          } else {
            range = `>= -(2${n9} ** ${(byteLength2 + 1) * 8 - 1}${n9}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n9}`;
          }
        } else {
          range = `>= ${min}${n9} and <= ${max}${n9}`;
        }
        throw new errors.ERR_OUT_OF_RANGE("value", range, value);
      }
      checkBounds(buf, offset, byteLength2);
    }
    function validateNumber(value, name3) {
      if (typeof value !== "number") {
        throw new errors.ERR_INVALID_ARG_TYPE(name3, "number", value);
      }
    }
    function boundsError(value, length, type2) {
      if (Math.floor(value) !== value) {
        validateNumber(value, type2);
        throw new errors.ERR_OUT_OF_RANGE(type2 || "offset", "an integer", value);
      }
      if (length < 0) {
        throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
      }
      throw new errors.ERR_OUT_OF_RANGE(
        type2 || "offset",
        `>= ${type2 ? 1 : 0} and <= ${length}`,
        value
      );
    }
    var INVALID_BASE64_RE = /[^+/0-9A-Za-z-_]/g;
    function base64clean(str) {
      str = str.split("=")[0];
      str = str.trim().replace(INVALID_BASE64_RE, "");
      if (str.length < 2)
        return "";
      while (str.length % 4 !== 0) {
        str = str + "=";
      }
      return str;
    }
    function utf8ToBytes(string, units) {
      units = units || Infinity;
      let codePoint;
      const length = string.length;
      let leadSurrogate = null;
      const bytes = [];
      for (let i7 = 0; i7 < length; ++i7) {
        codePoint = string.charCodeAt(i7);
        if (codePoint > 55295 && codePoint < 57344) {
          if (!leadSurrogate) {
            if (codePoint > 56319) {
              if ((units -= 3) > -1)
                bytes.push(239, 191, 189);
              continue;
            } else if (i7 + 1 === length) {
              if ((units -= 3) > -1)
                bytes.push(239, 191, 189);
              continue;
            }
            leadSurrogate = codePoint;
            continue;
          }
          if (codePoint < 56320) {
            if ((units -= 3) > -1)
              bytes.push(239, 191, 189);
            leadSurrogate = codePoint;
            continue;
          }
          codePoint = (leadSurrogate - 55296 << 10 | codePoint - 56320) + 65536;
        } else if (leadSurrogate) {
          if ((units -= 3) > -1)
            bytes.push(239, 191, 189);
        }
        leadSurrogate = null;
        if (codePoint < 128) {
          if ((units -= 1) < 0)
            break;
          bytes.push(codePoint);
        } else if (codePoint < 2048) {
          if ((units -= 2) < 0)
            break;
          bytes.push(
            codePoint >> 6 | 192,
            codePoint & 63 | 128
          );
        } else if (codePoint < 65536) {
          if ((units -= 3) < 0)
            break;
          bytes.push(
            codePoint >> 12 | 224,
            codePoint >> 6 & 63 | 128,
            codePoint & 63 | 128
          );
        } else if (codePoint < 1114112) {
          if ((units -= 4) < 0)
            break;
          bytes.push(
            codePoint >> 18 | 240,
            codePoint >> 12 & 63 | 128,
            codePoint >> 6 & 63 | 128,
            codePoint & 63 | 128
          );
        } else {
          throw new Error("Invalid code point");
        }
      }
      return bytes;
    }
    function asciiToBytes(str) {
      const byteArray = [];
      for (let i7 = 0; i7 < str.length; ++i7) {
        byteArray.push(str.charCodeAt(i7) & 255);
      }
      return byteArray;
    }
    function utf16leToBytes(str, units) {
      let c7, hi, lo;
      const byteArray = [];
      for (let i7 = 0; i7 < str.length; ++i7) {
        if ((units -= 2) < 0)
          break;
        c7 = str.charCodeAt(i7);
        hi = c7 >> 8;
        lo = c7 % 256;
        byteArray.push(lo);
        byteArray.push(hi);
      }
      return byteArray;
    }
    function base64ToBytes(str) {
      return base642.toByteArray(base64clean(str));
    }
    function blitBuffer(src, dst, offset, length) {
      let i7;
      for (i7 = 0; i7 < length; ++i7) {
        if (i7 + offset >= dst.length || i7 >= src.length)
          break;
        dst[i7 + offset] = src[i7];
      }
      return i7;
    }
    function isInstance(obj, type2) {
      return obj instanceof type2 || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type2.name;
    }
    function numberIsNaN(obj) {
      return obj !== obj;
    }
    var hexSliceLookupTable = function() {
      const alphabet = "0123456789abcdef";
      const table = new Array(256);
      for (let i7 = 0; i7 < 16; ++i7) {
        const i16 = i7 * 16;
        for (let j4 = 0; j4 < 16; ++j4) {
          table[i16 + j4] = alphabet[i7] + alphabet[j4];
        }
      }
      return table;
    }();
    function defineBigIntMethod(fn) {
      return typeof BigInt === "undefined" ? BufferBigIntNotDefined : fn;
    }
    function BufferBigIntNotDefined() {
      throw new Error("BigInt not supported");
    }
  }
});

// node_modules/object-sizeof/indexv2.js
var require_indexv2 = __commonJS({
  "node_modules/object-sizeof/indexv2.js"(exports10, module) {
    init_global();
    init_dirname();
    init_filename();
    init_buffer2();
    init_process2();
    var ECMA_SIZES = require_byte_size();
    var Buffer3 = require_buffer().Buffer;
    function preciseStringSizeNode(str) {
      return 12 + 4 * Math.ceil(str.length / 4);
    }
    function isNodeEnvironment() {
      if (typeof window !== "undefined" && typeof document !== "undefined") {
        return false;
      }
      return true;
    }
    function objectSizeComplex(obj) {
      let totalSize = 0;
      const errorIndication = -1;
      try {
        let potentialConversion = obj;
        if (obj instanceof Map) {
          potentialConversion = Object.fromEntries(obj);
        } else if (obj instanceof Set) {
          potentialConversion = Array.from(obj);
        }
        if (obj instanceof Int8Array) {
          return obj.length * ECMA_SIZES.Int8Array;
        } else if (obj instanceof Uint8Array || obj instanceof Uint8ClampedArray) {
          return obj.length * ECMA_SIZES.Uint8Array;
        } else if (obj instanceof Int16Array) {
          return obj.length * ECMA_SIZES.Int16Array;
        } else if (obj instanceof Uint16Array) {
          return obj.length * ECMA_SIZES.Uint16Array;
        } else if (obj instanceof Int32Array) {
          return obj.length * ECMA_SIZES.Int32Array;
        } else if (obj instanceof Uint32Array) {
          return obj.length * ECMA_SIZES.Uint32Array;
        } else if (obj instanceof Float32Array) {
          return obj.length * ECMA_SIZES.Float32Array;
        } else if (obj instanceof Float64Array) {
          return obj.length * ECMA_SIZES.Float64Array;
        }
        const objectToString = JSON.stringify(potentialConversion);
        const buffer2 = new Buffer3.from(objectToString);
        totalSize = buffer2.byteLength;
      } catch (ex) {
        console.error("Error detected, return " + errorIndication, ex);
        return errorIndication;
      }
      return totalSize;
    }
    function objectSizeSimple(obj) {
      const objectList = [];
      const stack = [obj];
      let bytes = 0;
      while (stack.length) {
        const value = stack.pop();
        if (typeof value === "boolean") {
          bytes += ECMA_SIZES.BYTES;
        } else if (typeof value === "string") {
          if (isNodeEnvironment()) {
            bytes += preciseStringSizeNode(value);
          } else {
            bytes += value.length * ECMA_SIZES.STRING;
          }
        } else if (typeof value === "number") {
          bytes += ECMA_SIZES.NUMBER;
        } else if (typeof value === "symbol") {
          const isGlobalSymbol = Symbol.keyFor && Symbol.keyFor(obj);
          if (isGlobalSymbol) {
            bytes += Symbol.keyFor(obj).length * ECMA_SIZES.STRING;
          } else {
            bytes += (obj.toString().length - 8) * ECMA_SIZES.STRING;
          }
        } else if (typeof value === "bigint") {
          bytes += Buffer3.from(value.toString()).byteLength;
        } else if (typeof value === "function") {
          bytes += value.toString().length;
        } else if (typeof value === "object" && objectList.indexOf(value) === -1) {
          objectList.push(value);
          for (const i7 in value) {
            stack.push(value[i7]);
          }
        }
      }
      return bytes;
    }
    module.exports = function(obj) {
      let totalSize = 0;
      if (obj !== null && typeof obj === "object") {
        totalSize = objectSizeComplex(obj);
      } else {
        totalSize = objectSizeSimple(obj);
      }
      return totalSize;
    };
  }
});

// src/index.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/TabbyAgent.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
init_events();

// node_modules/uuid/dist/esm-browser/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/uuid/dist/esm-browser/rng.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var getRandomValues;
var rnds8 = new Uint8Array(16);
function rng() {
  if (!getRandomValues) {
    getRandomValues = typeof crypto !== "undefined" && crypto.getRandomValues && crypto.getRandomValues.bind(crypto);
    if (!getRandomValues) {
      throw new Error("crypto.getRandomValues() not supported. See https://github.com/uuidjs/uuid#getrandomvalues-not-supported");
    }
  }
  return getRandomValues(rnds8);
}

// node_modules/uuid/dist/esm-browser/stringify.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var byteToHex = [];
for (let i7 = 0; i7 < 256; ++i7) {
  byteToHex.push((i7 + 256).toString(16).slice(1));
}
function unsafeStringify(arr, offset = 0) {
  return (byteToHex[arr[offset + 0]] + byteToHex[arr[offset + 1]] + byteToHex[arr[offset + 2]] + byteToHex[arr[offset + 3]] + "-" + byteToHex[arr[offset + 4]] + byteToHex[arr[offset + 5]] + "-" + byteToHex[arr[offset + 6]] + byteToHex[arr[offset + 7]] + "-" + byteToHex[arr[offset + 8]] + byteToHex[arr[offset + 9]] + "-" + byteToHex[arr[offset + 10]] + byteToHex[arr[offset + 11]] + byteToHex[arr[offset + 12]] + byteToHex[arr[offset + 13]] + byteToHex[arr[offset + 14]] + byteToHex[arr[offset + 15]]).toLowerCase();
}

// node_modules/uuid/dist/esm-browser/v4.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/uuid/dist/esm-browser/native.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var randomUUID = typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID.bind(crypto);
var native_default = {
  randomUUID
};

// node_modules/uuid/dist/esm-browser/v4.js
function v4(options, buf, offset) {
  if (native_default.randomUUID && !buf && !options) {
    return native_default.randomUUID();
  }
  options = options || {};
  const rnds = options.random || (options.rng || rng)();
  rnds[6] = rnds[6] & 15 | 64;
  rnds[8] = rnds[8] & 63 | 128;
  if (buf) {
    offset = offset || 0;
    for (let i7 = 0; i7 < 16; ++i7) {
      buf[offset + i7] = rnds[i7];
    }
    return buf;
  }
  return unsafeStringify(rnds);
}
var v4_default = v4;

// src/TabbyAgent.ts
var import_deep_equal = __toESM(require_deep_equal());
var import_deepmerge = __toESM(require_cjs());

// src/generated/index.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/generated/TabbyApi.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/generated/core/AxiosHttpRequest.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/generated/core/BaseHttpRequest.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var BaseHttpRequest = class {
  constructor(config2) {
    this.config = config2;
  }
};

// src/generated/core/request.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/axios.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/utils.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/bind.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function bind(fn, thisArg) {
  return function wrap() {
    return fn.apply(thisArg, arguments);
  };
}

// node_modules/axios/lib/utils.js
var { toString } = Object.prototype;
var { getPrototypeOf } = Object;
var kindOf = ((cache) => (thing) => {
  const str = toString.call(thing);
  return cache[str] || (cache[str] = str.slice(8, -1).toLowerCase());
})(/* @__PURE__ */ Object.create(null));
var kindOfTest = (type2) => {
  type2 = type2.toLowerCase();
  return (thing) => kindOf(thing) === type2;
};
var typeOfTest = (type2) => (thing) => typeof thing === type2;
var { isArray } = Array;
var isUndefined = typeOfTest("undefined");
function isBuffer(val) {
  return val !== null && !isUndefined(val) && val.constructor !== null && !isUndefined(val.constructor) && isFunction(val.constructor.isBuffer) && val.constructor.isBuffer(val);
}
var isArrayBuffer = kindOfTest("ArrayBuffer");
function isArrayBufferView(val) {
  let result;
  if (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView) {
    result = ArrayBuffer.isView(val);
  } else {
    result = val && val.buffer && isArrayBuffer(val.buffer);
  }
  return result;
}
var isString = typeOfTest("string");
var isFunction = typeOfTest("function");
var isNumber = typeOfTest("number");
var isObject = (thing) => thing !== null && typeof thing === "object";
var isBoolean = (thing) => thing === true || thing === false;
var isPlainObject = (val) => {
  if (kindOf(val) !== "object") {
    return false;
  }
  const prototype3 = getPrototypeOf(val);
  return (prototype3 === null || prototype3 === Object.prototype || Object.getPrototypeOf(prototype3) === null) && !(Symbol.toStringTag in val) && !(Symbol.iterator in val);
};
var isDate = kindOfTest("Date");
var isFile = kindOfTest("File");
var isBlob = kindOfTest("Blob");
var isFileList = kindOfTest("FileList");
var isStream = (val) => isObject(val) && isFunction(val.pipe);
var isFormData = (thing) => {
  let kind;
  return thing && (typeof FormData === "function" && thing instanceof FormData || isFunction(thing.append) && ((kind = kindOf(thing)) === "formdata" || // detect form-data instance
  kind === "object" && isFunction(thing.toString) && thing.toString() === "[object FormData]"));
};
var isURLSearchParams = kindOfTest("URLSearchParams");
var trim = (str) => str.trim ? str.trim() : str.replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, "");
function forEach(obj, fn, { allOwnKeys = false } = {}) {
  if (obj === null || typeof obj === "undefined") {
    return;
  }
  let i7;
  let l7;
  if (typeof obj !== "object") {
    obj = [obj];
  }
  if (isArray(obj)) {
    for (i7 = 0, l7 = obj.length; i7 < l7; i7++) {
      fn.call(null, obj[i7], i7, obj);
    }
  } else {
    const keys = allOwnKeys ? Object.getOwnPropertyNames(obj) : Object.keys(obj);
    const len = keys.length;
    let key;
    for (i7 = 0; i7 < len; i7++) {
      key = keys[i7];
      fn.call(null, obj[key], key, obj);
    }
  }
}
function findKey(obj, key) {
  key = key.toLowerCase();
  const keys = Object.keys(obj);
  let i7 = keys.length;
  let _key;
  while (i7-- > 0) {
    _key = keys[i7];
    if (key === _key.toLowerCase()) {
      return _key;
    }
  }
  return null;
}
var _global = (() => {
  if (typeof globalThis !== "undefined")
    return globalThis;
  return typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : global;
})();
var isContextDefined = (context) => !isUndefined(context) && context !== _global;
function merge() {
  const { caseless } = isContextDefined(this) && this || {};
  const result = {};
  const assignValue = (val, key) => {
    const targetKey = caseless && findKey(result, key) || key;
    if (isPlainObject(result[targetKey]) && isPlainObject(val)) {
      result[targetKey] = merge(result[targetKey], val);
    } else if (isPlainObject(val)) {
      result[targetKey] = merge({}, val);
    } else if (isArray(val)) {
      result[targetKey] = val.slice();
    } else {
      result[targetKey] = val;
    }
  };
  for (let i7 = 0, l7 = arguments.length; i7 < l7; i7++) {
    arguments[i7] && forEach(arguments[i7], assignValue);
  }
  return result;
}
var extend = (a7, b5, thisArg, { allOwnKeys } = {}) => {
  forEach(b5, (val, key) => {
    if (thisArg && isFunction(val)) {
      a7[key] = bind(val, thisArg);
    } else {
      a7[key] = val;
    }
  }, { allOwnKeys });
  return a7;
};
var stripBOM = (content) => {
  if (content.charCodeAt(0) === 65279) {
    content = content.slice(1);
  }
  return content;
};
var inherits = (constructor, superConstructor, props, descriptors2) => {
  constructor.prototype = Object.create(superConstructor.prototype, descriptors2);
  constructor.prototype.constructor = constructor;
  Object.defineProperty(constructor, "super", {
    value: superConstructor.prototype
  });
  props && Object.assign(constructor.prototype, props);
};
var toFlatObject = (sourceObj, destObj, filter2, propFilter) => {
  let props;
  let i7;
  let prop;
  const merged = {};
  destObj = destObj || {};
  if (sourceObj == null)
    return destObj;
  do {
    props = Object.getOwnPropertyNames(sourceObj);
    i7 = props.length;
    while (i7-- > 0) {
      prop = props[i7];
      if ((!propFilter || propFilter(prop, sourceObj, destObj)) && !merged[prop]) {
        destObj[prop] = sourceObj[prop];
        merged[prop] = true;
      }
    }
    sourceObj = filter2 !== false && getPrototypeOf(sourceObj);
  } while (sourceObj && (!filter2 || filter2(sourceObj, destObj)) && sourceObj !== Object.prototype);
  return destObj;
};
var endsWith = (str, searchString, position) => {
  str = String(str);
  if (position === void 0 || position > str.length) {
    position = str.length;
  }
  position -= searchString.length;
  const lastIndex = str.indexOf(searchString, position);
  return lastIndex !== -1 && lastIndex === position;
};
var toArray = (thing) => {
  if (!thing)
    return null;
  if (isArray(thing))
    return thing;
  let i7 = thing.length;
  if (!isNumber(i7))
    return null;
  const arr = new Array(i7);
  while (i7-- > 0) {
    arr[i7] = thing[i7];
  }
  return arr;
};
var isTypedArray = ((TypedArray) => {
  return (thing) => {
    return TypedArray && thing instanceof TypedArray;
  };
})(typeof Uint8Array !== "undefined" && getPrototypeOf(Uint8Array));
var forEachEntry = (obj, fn) => {
  const generator = obj && obj[Symbol.iterator];
  const iterator = generator.call(obj);
  let result;
  while ((result = iterator.next()) && !result.done) {
    const pair = result.value;
    fn.call(obj, pair[0], pair[1]);
  }
};
var matchAll = (regExp, str) => {
  let matches;
  const arr = [];
  while ((matches = regExp.exec(str)) !== null) {
    arr.push(matches);
  }
  return arr;
};
var isHTMLForm = kindOfTest("HTMLFormElement");
var toCamelCase = (str) => {
  return str.toLowerCase().replace(
    /[-_\s]([a-z\d])(\w*)/g,
    function replacer(m6, p1, p22) {
      return p1.toUpperCase() + p22;
    }
  );
};
var hasOwnProperty = (({ hasOwnProperty: hasOwnProperty2 }) => (obj, prop) => hasOwnProperty2.call(obj, prop))(Object.prototype);
var isRegExp = kindOfTest("RegExp");
var reduceDescriptors = (obj, reducer) => {
  const descriptors2 = Object.getOwnPropertyDescriptors(obj);
  const reducedDescriptors = {};
  forEach(descriptors2, (descriptor, name3) => {
    if (reducer(descriptor, name3, obj) !== false) {
      reducedDescriptors[name3] = descriptor;
    }
  });
  Object.defineProperties(obj, reducedDescriptors);
};
var freezeMethods = (obj) => {
  reduceDescriptors(obj, (descriptor, name3) => {
    if (isFunction(obj) && ["arguments", "caller", "callee"].indexOf(name3) !== -1) {
      return false;
    }
    const value = obj[name3];
    if (!isFunction(value))
      return;
    descriptor.enumerable = false;
    if ("writable" in descriptor) {
      descriptor.writable = false;
      return;
    }
    if (!descriptor.set) {
      descriptor.set = () => {
        throw Error("Can not rewrite read-only method '" + name3 + "'");
      };
    }
  });
};
var toObjectSet = (arrayOrString, delimiter2) => {
  const obj = {};
  const define2 = (arr) => {
    arr.forEach((value) => {
      obj[value] = true;
    });
  };
  isArray(arrayOrString) ? define2(arrayOrString) : define2(String(arrayOrString).split(delimiter2));
  return obj;
};
var noop2 = () => {
};
var toFiniteNumber = (value, defaultValue) => {
  value = +value;
  return Number.isFinite(value) ? value : defaultValue;
};
var ALPHA = "abcdefghijklmnopqrstuvwxyz";
var DIGIT = "0123456789";
var ALPHABET = {
  DIGIT,
  ALPHA,
  ALPHA_DIGIT: ALPHA + ALPHA.toUpperCase() + DIGIT
};
var generateString = (size = 16, alphabet = ALPHABET.ALPHA_DIGIT) => {
  let str = "";
  const { length } = alphabet;
  while (size--) {
    str += alphabet[Math.random() * length | 0];
  }
  return str;
};
function isSpecCompliantForm(thing) {
  return !!(thing && isFunction(thing.append) && thing[Symbol.toStringTag] === "FormData" && thing[Symbol.iterator]);
}
var toJSONObject = (obj) => {
  const stack = new Array(10);
  const visit = (source, i7) => {
    if (isObject(source)) {
      if (stack.indexOf(source) >= 0) {
        return;
      }
      if (!("toJSON" in source)) {
        stack[i7] = source;
        const target = isArray(source) ? [] : {};
        forEach(source, (value, key) => {
          const reducedValue = visit(value, i7 + 1);
          !isUndefined(reducedValue) && (target[key] = reducedValue);
        });
        stack[i7] = void 0;
        return target;
      }
    }
    return source;
  };
  return visit(obj, 0);
};
var isAsyncFn = kindOfTest("AsyncFunction");
var isThenable = (thing) => thing && (isObject(thing) || isFunction(thing)) && isFunction(thing.then) && isFunction(thing.catch);
var utils_default = {
  isArray,
  isArrayBuffer,
  isBuffer,
  isFormData,
  isArrayBufferView,
  isString,
  isNumber,
  isBoolean,
  isObject,
  isPlainObject,
  isUndefined,
  isDate,
  isFile,
  isBlob,
  isRegExp,
  isFunction,
  isStream,
  isURLSearchParams,
  isTypedArray,
  isFileList,
  forEach,
  merge,
  extend,
  trim,
  stripBOM,
  inherits,
  toFlatObject,
  kindOf,
  kindOfTest,
  endsWith,
  toArray,
  forEachEntry,
  matchAll,
  isHTMLForm,
  hasOwnProperty,
  hasOwnProp: hasOwnProperty,
  // an alias to avoid ESLint no-prototype-builtins detection
  reduceDescriptors,
  freezeMethods,
  toObjectSet,
  toCamelCase,
  noop: noop2,
  toFiniteNumber,
  findKey,
  global: _global,
  isContextDefined,
  ALPHABET,
  generateString,
  isSpecCompliantForm,
  toJSONObject,
  isAsyncFn,
  isThenable
};

// node_modules/axios/lib/core/Axios.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/buildURL.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/AxiosURLSearchParams.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/toFormData.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/core/AxiosError.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function AxiosError(message, code, config2, request2, response) {
  Error.call(this);
  if (Error.captureStackTrace) {
    Error.captureStackTrace(this, this.constructor);
  } else {
    this.stack = new Error().stack;
  }
  this.message = message;
  this.name = "AxiosError";
  code && (this.code = code);
  config2 && (this.config = config2);
  request2 && (this.request = request2);
  response && (this.response = response);
}
utils_default.inherits(AxiosError, Error, {
  toJSON: function toJSON() {
    return {
      // Standard
      message: this.message,
      name: this.name,
      // Microsoft
      description: this.description,
      number: this.number,
      // Mozilla
      fileName: this.fileName,
      lineNumber: this.lineNumber,
      columnNumber: this.columnNumber,
      stack: this.stack,
      // Axios
      config: utils_default.toJSONObject(this.config),
      code: this.code,
      status: this.response && this.response.status ? this.response.status : null
    };
  }
});
var prototype = AxiosError.prototype;
var descriptors = {};
[
  "ERR_BAD_OPTION_VALUE",
  "ERR_BAD_OPTION",
  "ECONNABORTED",
  "ETIMEDOUT",
  "ERR_NETWORK",
  "ERR_FR_TOO_MANY_REDIRECTS",
  "ERR_DEPRECATED",
  "ERR_BAD_RESPONSE",
  "ERR_BAD_REQUEST",
  "ERR_CANCELED",
  "ERR_NOT_SUPPORT",
  "ERR_INVALID_URL"
  // eslint-disable-next-line func-names
].forEach((code) => {
  descriptors[code] = { value: code };
});
Object.defineProperties(AxiosError, descriptors);
Object.defineProperty(prototype, "isAxiosError", { value: true });
AxiosError.from = (error, code, config2, request2, response, customProps) => {
  const axiosError = Object.create(prototype);
  utils_default.toFlatObject(error, axiosError, function filter2(obj) {
    return obj !== Error.prototype;
  }, (prop) => {
    return prop !== "isAxiosError";
  });
  AxiosError.call(axiosError, error.message, code, config2, request2, response);
  axiosError.cause = error;
  axiosError.name = error.name;
  customProps && Object.assign(axiosError, customProps);
  return axiosError;
};
var AxiosError_default = AxiosError;

// node_modules/axios/lib/helpers/null.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var null_default = null;

// node_modules/axios/lib/helpers/toFormData.js
function isVisitable(thing) {
  return utils_default.isPlainObject(thing) || utils_default.isArray(thing);
}
function removeBrackets(key) {
  return utils_default.endsWith(key, "[]") ? key.slice(0, -2) : key;
}
function renderKey(path2, key, dots) {
  if (!path2)
    return key;
  return path2.concat(key).map(function each(token, i7) {
    token = removeBrackets(token);
    return !dots && i7 ? "[" + token + "]" : token;
  }).join(dots ? "." : "");
}
function isFlatArray(arr) {
  return utils_default.isArray(arr) && !arr.some(isVisitable);
}
var predicates = utils_default.toFlatObject(utils_default, {}, null, function filter(prop) {
  return /^is[A-Z]/.test(prop);
});
function toFormData(obj, formData, options) {
  if (!utils_default.isObject(obj)) {
    throw new TypeError("target must be an object");
  }
  formData = formData || new (FormData)();
  options = utils_default.toFlatObject(options, {
    metaTokens: true,
    dots: false,
    indexes: false
  }, false, function defined(option, source) {
    return !utils_default.isUndefined(source[option]);
  });
  const metaTokens = options.metaTokens;
  const visitor = options.visitor || defaultVisitor;
  const dots = options.dots;
  const indexes = options.indexes;
  const _Blob = options.Blob || typeof Blob !== "undefined" && Blob;
  const useBlob = _Blob && utils_default.isSpecCompliantForm(formData);
  if (!utils_default.isFunction(visitor)) {
    throw new TypeError("visitor must be a function");
  }
  function convertValue(value) {
    if (value === null)
      return "";
    if (utils_default.isDate(value)) {
      return value.toISOString();
    }
    if (!useBlob && utils_default.isBlob(value)) {
      throw new AxiosError_default("Blob is not supported. Use a Buffer instead.");
    }
    if (utils_default.isArrayBuffer(value) || utils_default.isTypedArray(value)) {
      return useBlob && typeof Blob === "function" ? new Blob([value]) : Buffer2.from(value);
    }
    return value;
  }
  function defaultVisitor(value, key, path2) {
    let arr = value;
    if (value && !path2 && typeof value === "object") {
      if (utils_default.endsWith(key, "{}")) {
        key = metaTokens ? key : key.slice(0, -2);
        value = JSON.stringify(value);
      } else if (utils_default.isArray(value) && isFlatArray(value) || (utils_default.isFileList(value) || utils_default.endsWith(key, "[]")) && (arr = utils_default.toArray(value))) {
        key = removeBrackets(key);
        arr.forEach(function each(el, index) {
          !(utils_default.isUndefined(el) || el === null) && formData.append(
            // eslint-disable-next-line no-nested-ternary
            indexes === true ? renderKey([key], index, dots) : indexes === null ? key : key + "[]",
            convertValue(el)
          );
        });
        return false;
      }
    }
    if (isVisitable(value)) {
      return true;
    }
    formData.append(renderKey(path2, key, dots), convertValue(value));
    return false;
  }
  const stack = [];
  const exposedHelpers = Object.assign(predicates, {
    defaultVisitor,
    convertValue,
    isVisitable
  });
  function build(value, path2) {
    if (utils_default.isUndefined(value))
      return;
    if (stack.indexOf(value) !== -1) {
      throw Error("Circular reference detected in " + path2.join("."));
    }
    stack.push(value);
    utils_default.forEach(value, function each(el, key) {
      const result = !(utils_default.isUndefined(el) || el === null) && visitor.call(
        formData,
        el,
        utils_default.isString(key) ? key.trim() : key,
        path2,
        exposedHelpers
      );
      if (result === true) {
        build(el, path2 ? path2.concat(key) : [key]);
      }
    });
    stack.pop();
  }
  if (!utils_default.isObject(obj)) {
    throw new TypeError("data must be an object");
  }
  build(obj);
  return formData;
}
var toFormData_default = toFormData;

// node_modules/axios/lib/helpers/AxiosURLSearchParams.js
function encode(str) {
  const charMap = {
    "!": "%21",
    "'": "%27",
    "(": "%28",
    ")": "%29",
    "~": "%7E",
    "%20": "+",
    "%00": "\0"
  };
  return encodeURIComponent(str).replace(/[!'()~]|%20|%00/g, function replacer(match) {
    return charMap[match];
  });
}
function AxiosURLSearchParams(params, options) {
  this._pairs = [];
  params && toFormData_default(params, this, options);
}
var prototype2 = AxiosURLSearchParams.prototype;
prototype2.append = function append(name3, value) {
  this._pairs.push([name3, value]);
};
prototype2.toString = function toString2(encoder) {
  const _encode = encoder ? function(value) {
    return encoder.call(this, value, encode);
  } : encode;
  return this._pairs.map(function each(pair) {
    return _encode(pair[0]) + "=" + _encode(pair[1]);
  }, "").join("&");
};
var AxiosURLSearchParams_default = AxiosURLSearchParams;

// node_modules/axios/lib/helpers/buildURL.js
function encode2(val) {
  return encodeURIComponent(val).replace(/%3A/gi, ":").replace(/%24/g, "$").replace(/%2C/gi, ",").replace(/%20/g, "+").replace(/%5B/gi, "[").replace(/%5D/gi, "]");
}
function buildURL(url, params, options) {
  if (!params) {
    return url;
  }
  const _encode = options && options.encode || encode2;
  const serializeFn = options && options.serialize;
  let serializedParams;
  if (serializeFn) {
    serializedParams = serializeFn(params, options);
  } else {
    serializedParams = utils_default.isURLSearchParams(params) ? params.toString() : new AxiosURLSearchParams_default(params, options).toString(_encode);
  }
  if (serializedParams) {
    const hashmarkIndex = url.indexOf("#");
    if (hashmarkIndex !== -1) {
      url = url.slice(0, hashmarkIndex);
    }
    url += (url.indexOf("?") === -1 ? "?" : "&") + serializedParams;
  }
  return url;
}

// node_modules/axios/lib/core/InterceptorManager.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var InterceptorManager = class {
  constructor() {
    this.handlers = [];
  }
  /**
   * Add a new interceptor to the stack
   *
   * @param {Function} fulfilled The function to handle `then` for a `Promise`
   * @param {Function} rejected The function to handle `reject` for a `Promise`
   *
   * @return {Number} An ID used to remove interceptor later
   */
  use(fulfilled, rejected, options) {
    this.handlers.push({
      fulfilled,
      rejected,
      synchronous: options ? options.synchronous : false,
      runWhen: options ? options.runWhen : null
    });
    return this.handlers.length - 1;
  }
  /**
   * Remove an interceptor from the stack
   *
   * @param {Number} id The ID that was returned by `use`
   *
   * @returns {Boolean} `true` if the interceptor was removed, `false` otherwise
   */
  eject(id) {
    if (this.handlers[id]) {
      this.handlers[id] = null;
    }
  }
  /**
   * Clear all interceptors from the stack
   *
   * @returns {void}
   */
  clear() {
    if (this.handlers) {
      this.handlers = [];
    }
  }
  /**
   * Iterate over all the registered interceptors
   *
   * This method is particularly useful for skipping over any
   * interceptors that may have become `null` calling `eject`.
   *
   * @param {Function} fn The function to call for each interceptor
   *
   * @returns {void}
   */
  forEach(fn) {
    utils_default.forEach(this.handlers, function forEachHandler(h8) {
      if (h8 !== null) {
        fn(h8);
      }
    });
  }
};
var InterceptorManager_default = InterceptorManager;

// node_modules/axios/lib/core/dispatchRequest.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/core/transformData.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/defaults/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/defaults/transitional.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var transitional_default = {
  silentJSONParsing: true,
  forcedJSONParsing: true,
  clarifyTimeoutError: false
};

// node_modules/axios/lib/helpers/toURLEncodedForm.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/platform/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/platform/browser/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/platform/browser/classes/URLSearchParams.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var URLSearchParams_default = typeof URLSearchParams !== "undefined" ? URLSearchParams : AxiosURLSearchParams_default;

// node_modules/axios/lib/platform/browser/classes/FormData.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var FormData_default = typeof FormData !== "undefined" ? FormData : null;

// node_modules/axios/lib/platform/browser/classes/Blob.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var Blob_default = typeof Blob !== "undefined" ? Blob : null;

// node_modules/axios/lib/platform/browser/index.js
var isStandardBrowserEnv = (() => {
  let product;
  if (typeof navigator !== "undefined" && ((product = navigator.product) === "ReactNative" || product === "NativeScript" || product === "NS")) {
    return false;
  }
  return typeof window !== "undefined" && typeof document !== "undefined";
})();
var isStandardBrowserWebWorkerEnv = (() => {
  return typeof WorkerGlobalScope !== "undefined" && // eslint-disable-next-line no-undef
  self instanceof WorkerGlobalScope && typeof self.importScripts === "function";
})();
var browser_default = {
  isBrowser: true,
  classes: {
    URLSearchParams: URLSearchParams_default,
    FormData: FormData_default,
    Blob: Blob_default
  },
  isStandardBrowserEnv,
  isStandardBrowserWebWorkerEnv,
  protocols: ["http", "https", "file", "blob", "url", "data"]
};

// node_modules/axios/lib/helpers/toURLEncodedForm.js
function toURLEncodedForm(data, options) {
  return toFormData_default(data, new browser_default.classes.URLSearchParams(), Object.assign({
    visitor: function(value, key, path2, helpers) {
      if (browser_default.isNode && utils_default.isBuffer(value)) {
        this.append(key, value.toString("base64"));
        return false;
      }
      return helpers.defaultVisitor.apply(this, arguments);
    }
  }, options));
}

// node_modules/axios/lib/helpers/formDataToJSON.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function parsePropPath(name3) {
  return utils_default.matchAll(/\w+|\[(\w*)]/g, name3).map((match) => {
    return match[0] === "[]" ? "" : match[1] || match[0];
  });
}
function arrayToObject(arr) {
  const obj = {};
  const keys = Object.keys(arr);
  let i7;
  const len = keys.length;
  let key;
  for (i7 = 0; i7 < len; i7++) {
    key = keys[i7];
    obj[key] = arr[key];
  }
  return obj;
}
function formDataToJSON(formData) {
  function buildPath(path2, value, target, index) {
    let name3 = path2[index++];
    const isNumericKey = Number.isFinite(+name3);
    const isLast = index >= path2.length;
    name3 = !name3 && utils_default.isArray(target) ? target.length : name3;
    if (isLast) {
      if (utils_default.hasOwnProp(target, name3)) {
        target[name3] = [target[name3], value];
      } else {
        target[name3] = value;
      }
      return !isNumericKey;
    }
    if (!target[name3] || !utils_default.isObject(target[name3])) {
      target[name3] = [];
    }
    const result = buildPath(path2, value, target[name3], index);
    if (result && utils_default.isArray(target[name3])) {
      target[name3] = arrayToObject(target[name3]);
    }
    return !isNumericKey;
  }
  if (utils_default.isFormData(formData) && utils_default.isFunction(formData.entries)) {
    const obj = {};
    utils_default.forEachEntry(formData, (name3, value) => {
      buildPath(parsePropPath(name3), value, obj, 0);
    });
    return obj;
  }
  return null;
}
var formDataToJSON_default = formDataToJSON;

// node_modules/axios/lib/defaults/index.js
var DEFAULT_CONTENT_TYPE = {
  "Content-Type": void 0
};
function stringifySafely(rawValue, parser, encoder) {
  if (utils_default.isString(rawValue)) {
    try {
      (parser || JSON.parse)(rawValue);
      return utils_default.trim(rawValue);
    } catch (e10) {
      if (e10.name !== "SyntaxError") {
        throw e10;
      }
    }
  }
  return (encoder || JSON.stringify)(rawValue);
}
var defaults = {
  transitional: transitional_default,
  adapter: ["xhr", "http"],
  transformRequest: [function transformRequest(data, headers) {
    const contentType = headers.getContentType() || "";
    const hasJSONContentType = contentType.indexOf("application/json") > -1;
    const isObjectPayload = utils_default.isObject(data);
    if (isObjectPayload && utils_default.isHTMLForm(data)) {
      data = new FormData(data);
    }
    const isFormData3 = utils_default.isFormData(data);
    if (isFormData3) {
      if (!hasJSONContentType) {
        return data;
      }
      return hasJSONContentType ? JSON.stringify(formDataToJSON_default(data)) : data;
    }
    if (utils_default.isArrayBuffer(data) || utils_default.isBuffer(data) || utils_default.isStream(data) || utils_default.isFile(data) || utils_default.isBlob(data)) {
      return data;
    }
    if (utils_default.isArrayBufferView(data)) {
      return data.buffer;
    }
    if (utils_default.isURLSearchParams(data)) {
      headers.setContentType("application/x-www-form-urlencoded;charset=utf-8", false);
      return data.toString();
    }
    let isFileList2;
    if (isObjectPayload) {
      if (contentType.indexOf("application/x-www-form-urlencoded") > -1) {
        return toURLEncodedForm(data, this.formSerializer).toString();
      }
      if ((isFileList2 = utils_default.isFileList(data)) || contentType.indexOf("multipart/form-data") > -1) {
        const _FormData = this.env && this.env.FormData;
        return toFormData_default(
          isFileList2 ? { "files[]": data } : data,
          _FormData && new _FormData(),
          this.formSerializer
        );
      }
    }
    if (isObjectPayload || hasJSONContentType) {
      headers.setContentType("application/json", false);
      return stringifySafely(data);
    }
    return data;
  }],
  transformResponse: [function transformResponse(data) {
    const transitional2 = this.transitional || defaults.transitional;
    const forcedJSONParsing = transitional2 && transitional2.forcedJSONParsing;
    const JSONRequested = this.responseType === "json";
    if (data && utils_default.isString(data) && (forcedJSONParsing && !this.responseType || JSONRequested)) {
      const silentJSONParsing = transitional2 && transitional2.silentJSONParsing;
      const strictJSONParsing = !silentJSONParsing && JSONRequested;
      try {
        return JSON.parse(data);
      } catch (e10) {
        if (strictJSONParsing) {
          if (e10.name === "SyntaxError") {
            throw AxiosError_default.from(e10, AxiosError_default.ERR_BAD_RESPONSE, this, null, this.response);
          }
          throw e10;
        }
      }
    }
    return data;
  }],
  /**
   * A timeout in milliseconds to abort a request. If set to 0 (default) a
   * timeout is not created.
   */
  timeout: 0,
  xsrfCookieName: "XSRF-TOKEN",
  xsrfHeaderName: "X-XSRF-TOKEN",
  maxContentLength: -1,
  maxBodyLength: -1,
  env: {
    FormData: browser_default.classes.FormData,
    Blob: browser_default.classes.Blob
  },
  validateStatus: function validateStatus(status) {
    return status >= 200 && status < 300;
  },
  headers: {
    common: {
      "Accept": "application/json, text/plain, */*"
    }
  }
};
utils_default.forEach(["delete", "get", "head"], function forEachMethodNoData(method) {
  defaults.headers[method] = {};
});
utils_default.forEach(["post", "put", "patch"], function forEachMethodWithData(method) {
  defaults.headers[method] = utils_default.merge(DEFAULT_CONTENT_TYPE);
});
var defaults_default = defaults;

// node_modules/axios/lib/core/AxiosHeaders.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/parseHeaders.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var ignoreDuplicateOf = utils_default.toObjectSet([
  "age",
  "authorization",
  "content-length",
  "content-type",
  "etag",
  "expires",
  "from",
  "host",
  "if-modified-since",
  "if-unmodified-since",
  "last-modified",
  "location",
  "max-forwards",
  "proxy-authorization",
  "referer",
  "retry-after",
  "user-agent"
]);
var parseHeaders_default = (rawHeaders) => {
  const parsed = {};
  let key;
  let val;
  let i7;
  rawHeaders && rawHeaders.split("\n").forEach(function parser(line) {
    i7 = line.indexOf(":");
    key = line.substring(0, i7).trim().toLowerCase();
    val = line.substring(i7 + 1).trim();
    if (!key || parsed[key] && ignoreDuplicateOf[key]) {
      return;
    }
    if (key === "set-cookie") {
      if (parsed[key]) {
        parsed[key].push(val);
      } else {
        parsed[key] = [val];
      }
    } else {
      parsed[key] = parsed[key] ? parsed[key] + ", " + val : val;
    }
  });
  return parsed;
};

// node_modules/axios/lib/core/AxiosHeaders.js
var $internals = Symbol("internals");
function normalizeHeader(header) {
  return header && String(header).trim().toLowerCase();
}
function normalizeValue(value) {
  if (value === false || value == null) {
    return value;
  }
  return utils_default.isArray(value) ? value.map(normalizeValue) : String(value);
}
function parseTokens(str) {
  const tokens = /* @__PURE__ */ Object.create(null);
  const tokensRE = /([^\s,;=]+)\s*(?:=\s*([^,;]+))?/g;
  let match;
  while (match = tokensRE.exec(str)) {
    tokens[match[1]] = match[2];
  }
  return tokens;
}
var isValidHeaderName = (str) => /^[-_a-zA-Z0-9^`|~,!#$%&'*+.]+$/.test(str.trim());
function matchHeaderValue(context, value, header, filter2, isHeaderNameFilter) {
  if (utils_default.isFunction(filter2)) {
    return filter2.call(this, value, header);
  }
  if (isHeaderNameFilter) {
    value = header;
  }
  if (!utils_default.isString(value))
    return;
  if (utils_default.isString(filter2)) {
    return value.indexOf(filter2) !== -1;
  }
  if (utils_default.isRegExp(filter2)) {
    return filter2.test(value);
  }
}
function formatHeader(header) {
  return header.trim().toLowerCase().replace(/([a-z\d])(\w*)/g, (w4, char, str) => {
    return char.toUpperCase() + str;
  });
}
function buildAccessors(obj, header) {
  const accessorName = utils_default.toCamelCase(" " + header);
  ["get", "set", "has"].forEach((methodName) => {
    Object.defineProperty(obj, methodName + accessorName, {
      value: function(arg1, arg2, arg3) {
        return this[methodName].call(this, header, arg1, arg2, arg3);
      },
      configurable: true
    });
  });
}
var AxiosHeaders = class {
  constructor(headers) {
    headers && this.set(headers);
  }
  set(header, valueOrRewrite, rewrite) {
    const self2 = this;
    function setHeader(_value, _header, _rewrite) {
      const lHeader = normalizeHeader(_header);
      if (!lHeader) {
        throw new Error("header name must be a non-empty string");
      }
      const key = utils_default.findKey(self2, lHeader);
      if (!key || self2[key] === void 0 || _rewrite === true || _rewrite === void 0 && self2[key] !== false) {
        self2[key || _header] = normalizeValue(_value);
      }
    }
    const setHeaders = (headers, _rewrite) => utils_default.forEach(headers, (_value, _header) => setHeader(_value, _header, _rewrite));
    if (utils_default.isPlainObject(header) || header instanceof this.constructor) {
      setHeaders(header, valueOrRewrite);
    } else if (utils_default.isString(header) && (header = header.trim()) && !isValidHeaderName(header)) {
      setHeaders(parseHeaders_default(header), valueOrRewrite);
    } else {
      header != null && setHeader(valueOrRewrite, header, rewrite);
    }
    return this;
  }
  get(header, parser) {
    header = normalizeHeader(header);
    if (header) {
      const key = utils_default.findKey(this, header);
      if (key) {
        const value = this[key];
        if (!parser) {
          return value;
        }
        if (parser === true) {
          return parseTokens(value);
        }
        if (utils_default.isFunction(parser)) {
          return parser.call(this, value, key);
        }
        if (utils_default.isRegExp(parser)) {
          return parser.exec(value);
        }
        throw new TypeError("parser must be boolean|regexp|function");
      }
    }
  }
  has(header, matcher) {
    header = normalizeHeader(header);
    if (header) {
      const key = utils_default.findKey(this, header);
      return !!(key && this[key] !== void 0 && (!matcher || matchHeaderValue(this, this[key], key, matcher)));
    }
    return false;
  }
  delete(header, matcher) {
    const self2 = this;
    let deleted = false;
    function deleteHeader(_header) {
      _header = normalizeHeader(_header);
      if (_header) {
        const key = utils_default.findKey(self2, _header);
        if (key && (!matcher || matchHeaderValue(self2, self2[key], key, matcher))) {
          delete self2[key];
          deleted = true;
        }
      }
    }
    if (utils_default.isArray(header)) {
      header.forEach(deleteHeader);
    } else {
      deleteHeader(header);
    }
    return deleted;
  }
  clear(matcher) {
    const keys = Object.keys(this);
    let i7 = keys.length;
    let deleted = false;
    while (i7--) {
      const key = keys[i7];
      if (!matcher || matchHeaderValue(this, this[key], key, matcher, true)) {
        delete this[key];
        deleted = true;
      }
    }
    return deleted;
  }
  normalize(format5) {
    const self2 = this;
    const headers = {};
    utils_default.forEach(this, (value, header) => {
      const key = utils_default.findKey(headers, header);
      if (key) {
        self2[key] = normalizeValue(value);
        delete self2[header];
        return;
      }
      const normalized = format5 ? formatHeader(header) : String(header).trim();
      if (normalized !== header) {
        delete self2[header];
      }
      self2[normalized] = normalizeValue(value);
      headers[normalized] = true;
    });
    return this;
  }
  concat(...targets) {
    return this.constructor.concat(this, ...targets);
  }
  toJSON(asStrings) {
    const obj = /* @__PURE__ */ Object.create(null);
    utils_default.forEach(this, (value, header) => {
      value != null && value !== false && (obj[header] = asStrings && utils_default.isArray(value) ? value.join(", ") : value);
    });
    return obj;
  }
  [Symbol.iterator]() {
    return Object.entries(this.toJSON())[Symbol.iterator]();
  }
  toString() {
    return Object.entries(this.toJSON()).map(([header, value]) => header + ": " + value).join("\n");
  }
  get [Symbol.toStringTag]() {
    return "AxiosHeaders";
  }
  static from(thing) {
    return thing instanceof this ? thing : new this(thing);
  }
  static concat(first, ...targets) {
    const computed = new this(first);
    targets.forEach((target) => computed.set(target));
    return computed;
  }
  static accessor(header) {
    const internals = this[$internals] = this[$internals] = {
      accessors: {}
    };
    const accessors = internals.accessors;
    const prototype3 = this.prototype;
    function defineAccessor(_header) {
      const lHeader = normalizeHeader(_header);
      if (!accessors[lHeader]) {
        buildAccessors(prototype3, _header);
        accessors[lHeader] = true;
      }
    }
    utils_default.isArray(header) ? header.forEach(defineAccessor) : defineAccessor(header);
    return this;
  }
};
AxiosHeaders.accessor(["Content-Type", "Content-Length", "Accept", "Accept-Encoding", "User-Agent", "Authorization"]);
utils_default.freezeMethods(AxiosHeaders.prototype);
utils_default.freezeMethods(AxiosHeaders);
var AxiosHeaders_default = AxiosHeaders;

// node_modules/axios/lib/core/transformData.js
function transformData(fns, response) {
  const config2 = this || defaults_default;
  const context = response || config2;
  const headers = AxiosHeaders_default.from(context.headers);
  let data = context.data;
  utils_default.forEach(fns, function transform(fn) {
    data = fn.call(config2, data, headers.normalize(), response ? response.status : void 0);
  });
  headers.normalize();
  return data;
}

// node_modules/axios/lib/cancel/isCancel.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function isCancel(value) {
  return !!(value && value.__CANCEL__);
}

// node_modules/axios/lib/cancel/CanceledError.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function CanceledError(message, config2, request2) {
  AxiosError_default.call(this, message == null ? "canceled" : message, AxiosError_default.ERR_CANCELED, config2, request2);
  this.name = "CanceledError";
}
utils_default.inherits(CanceledError, AxiosError_default, {
  __CANCEL__: true
});
var CanceledError_default = CanceledError;

// node_modules/axios/lib/adapters/adapters.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/adapters/xhr.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/core/settle.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function settle(resolve4, reject, response) {
  const validateStatus2 = response.config.validateStatus;
  if (!response.status || !validateStatus2 || validateStatus2(response.status)) {
    resolve4(response);
  } else {
    reject(new AxiosError_default(
      "Request failed with status code " + response.status,
      [AxiosError_default.ERR_BAD_REQUEST, AxiosError_default.ERR_BAD_RESPONSE][Math.floor(response.status / 100) - 4],
      response.config,
      response.request,
      response
    ));
  }
}

// node_modules/axios/lib/helpers/cookies.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var cookies_default = browser_default.isStandardBrowserEnv ? (
  // Standard browser envs support document.cookie
  function standardBrowserEnv() {
    return {
      write: function write2(name3, value, expires, path2, domain2, secure) {
        const cookie = [];
        cookie.push(name3 + "=" + encodeURIComponent(value));
        if (utils_default.isNumber(expires)) {
          cookie.push("expires=" + new Date(expires).toGMTString());
        }
        if (utils_default.isString(path2)) {
          cookie.push("path=" + path2);
        }
        if (utils_default.isString(domain2)) {
          cookie.push("domain=" + domain2);
        }
        if (secure === true) {
          cookie.push("secure");
        }
        document.cookie = cookie.join("; ");
      },
      read: function read2(name3) {
        const match = document.cookie.match(new RegExp("(^|;\\s*)(" + name3 + ")=([^;]*)"));
        return match ? decodeURIComponent(match[3]) : null;
      },
      remove: function remove(name3) {
        this.write(name3, "", Date.now() - 864e5);
      }
    };
  }()
) : (
  // Non standard browser env (web workers, react-native) lack needed support.
  function nonStandardBrowserEnv() {
    return {
      write: function write2() {
      },
      read: function read2() {
        return null;
      },
      remove: function remove() {
      }
    };
  }()
);

// node_modules/axios/lib/core/buildFullPath.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/helpers/isAbsoluteURL.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function isAbsoluteURL(url) {
  return /^([a-z][a-z\d+\-.]*:)?\/\//i.test(url);
}

// node_modules/axios/lib/helpers/combineURLs.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function combineURLs(baseURL, relativeURL) {
  return relativeURL ? baseURL.replace(/\/+$/, "") + "/" + relativeURL.replace(/^\/+/, "") : baseURL;
}

// node_modules/axios/lib/core/buildFullPath.js
function buildFullPath(baseURL, requestedURL) {
  if (baseURL && !isAbsoluteURL(requestedURL)) {
    return combineURLs(baseURL, requestedURL);
  }
  return requestedURL;
}

// node_modules/axios/lib/helpers/isURLSameOrigin.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var isURLSameOrigin_default = browser_default.isStandardBrowserEnv ? (
  // Standard browser envs have full support of the APIs needed to test
  // whether the request URL is of the same origin as current location.
  function standardBrowserEnv2() {
    const msie = /(msie|trident)/i.test(navigator.userAgent);
    const urlParsingNode = document.createElement("a");
    let originURL;
    function resolveURL(url) {
      let href = url;
      if (msie) {
        urlParsingNode.setAttribute("href", href);
        href = urlParsingNode.href;
      }
      urlParsingNode.setAttribute("href", href);
      return {
        href: urlParsingNode.href,
        protocol: urlParsingNode.protocol ? urlParsingNode.protocol.replace(/:$/, "") : "",
        host: urlParsingNode.host,
        search: urlParsingNode.search ? urlParsingNode.search.replace(/^\?/, "") : "",
        hash: urlParsingNode.hash ? urlParsingNode.hash.replace(/^#/, "") : "",
        hostname: urlParsingNode.hostname,
        port: urlParsingNode.port,
        pathname: urlParsingNode.pathname.charAt(0) === "/" ? urlParsingNode.pathname : "/" + urlParsingNode.pathname
      };
    }
    originURL = resolveURL(window.location.href);
    return function isURLSameOrigin(requestURL) {
      const parsed = utils_default.isString(requestURL) ? resolveURL(requestURL) : requestURL;
      return parsed.protocol === originURL.protocol && parsed.host === originURL.host;
    };
  }()
) : (
  // Non standard browser envs (web workers, react-native) lack needed support.
  function nonStandardBrowserEnv2() {
    return function isURLSameOrigin() {
      return true;
    };
  }()
);

// node_modules/axios/lib/helpers/parseProtocol.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function parseProtocol(url) {
  const match = /^([-+\w]{1,25})(:?\/\/|:)/.exec(url);
  return match && match[1] || "";
}

// node_modules/axios/lib/helpers/speedometer.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function speedometer(samplesCount, min) {
  samplesCount = samplesCount || 10;
  const bytes = new Array(samplesCount);
  const timestamps = new Array(samplesCount);
  let head = 0;
  let tail = 0;
  let firstSampleTS;
  min = min !== void 0 ? min : 1e3;
  return function push(chunkLength) {
    const now = Date.now();
    const startedAt = timestamps[tail];
    if (!firstSampleTS) {
      firstSampleTS = now;
    }
    bytes[head] = chunkLength;
    timestamps[head] = now;
    let i7 = tail;
    let bytesCount = 0;
    while (i7 !== head) {
      bytesCount += bytes[i7++];
      i7 = i7 % samplesCount;
    }
    head = (head + 1) % samplesCount;
    if (head === tail) {
      tail = (tail + 1) % samplesCount;
    }
    if (now - firstSampleTS < min) {
      return;
    }
    const passed = startedAt && now - startedAt;
    return passed ? Math.round(bytesCount * 1e3 / passed) : void 0;
  };
}
var speedometer_default = speedometer;

// node_modules/axios/lib/adapters/xhr.js
function progressEventReducer(listener, isDownloadStream) {
  let bytesNotified = 0;
  const _speedometer = speedometer_default(50, 250);
  return (e10) => {
    const loaded = e10.loaded;
    const total = e10.lengthComputable ? e10.total : void 0;
    const progressBytes = loaded - bytesNotified;
    const rate = _speedometer(progressBytes);
    const inRange = loaded <= total;
    bytesNotified = loaded;
    const data = {
      loaded,
      total,
      progress: total ? loaded / total : void 0,
      bytes: progressBytes,
      rate: rate ? rate : void 0,
      estimated: rate && total && inRange ? (total - loaded) / rate : void 0,
      event: e10
    };
    data[isDownloadStream ? "download" : "upload"] = true;
    listener(data);
  };
}
var isXHRAdapterSupported = typeof XMLHttpRequest !== "undefined";
var xhr_default = isXHRAdapterSupported && function(config2) {
  return new Promise(function dispatchXhrRequest(resolve4, reject) {
    let requestData = config2.data;
    const requestHeaders = AxiosHeaders_default.from(config2.headers).normalize();
    const responseType = config2.responseType;
    let onCanceled;
    function done() {
      if (config2.cancelToken) {
        config2.cancelToken.unsubscribe(onCanceled);
      }
      if (config2.signal) {
        config2.signal.removeEventListener("abort", onCanceled);
      }
    }
    if (utils_default.isFormData(requestData)) {
      if (browser_default.isStandardBrowserEnv || browser_default.isStandardBrowserWebWorkerEnv) {
        requestHeaders.setContentType(false);
      } else {
        requestHeaders.setContentType("multipart/form-data;", false);
      }
    }
    let request2 = new XMLHttpRequest();
    if (config2.auth) {
      const username = config2.auth.username || "";
      const password = config2.auth.password ? unescape(encodeURIComponent(config2.auth.password)) : "";
      requestHeaders.set("Authorization", "Basic " + btoa(username + ":" + password));
    }
    const fullPath = buildFullPath(config2.baseURL, config2.url);
    request2.open(config2.method.toUpperCase(), buildURL(fullPath, config2.params, config2.paramsSerializer), true);
    request2.timeout = config2.timeout;
    function onloadend() {
      if (!request2) {
        return;
      }
      const responseHeaders = AxiosHeaders_default.from(
        "getAllResponseHeaders" in request2 && request2.getAllResponseHeaders()
      );
      const responseData = !responseType || responseType === "text" || responseType === "json" ? request2.responseText : request2.response;
      const response = {
        data: responseData,
        status: request2.status,
        statusText: request2.statusText,
        headers: responseHeaders,
        config: config2,
        request: request2
      };
      settle(function _resolve2(value) {
        resolve4(value);
        done();
      }, function _reject2(err) {
        reject(err);
        done();
      }, response);
      request2 = null;
    }
    if ("onloadend" in request2) {
      request2.onloadend = onloadend;
    } else {
      request2.onreadystatechange = function handleLoad() {
        if (!request2 || request2.readyState !== 4) {
          return;
        }
        if (request2.status === 0 && !(request2.responseURL && request2.responseURL.indexOf("file:") === 0)) {
          return;
        }
        setTimeout(onloadend);
      };
    }
    request2.onabort = function handleAbort() {
      if (!request2) {
        return;
      }
      reject(new AxiosError_default("Request aborted", AxiosError_default.ECONNABORTED, config2, request2));
      request2 = null;
    };
    request2.onerror = function handleError() {
      reject(new AxiosError_default("Network Error", AxiosError_default.ERR_NETWORK, config2, request2));
      request2 = null;
    };
    request2.ontimeout = function handleTimeout() {
      let timeoutErrorMessage = config2.timeout ? "timeout of " + config2.timeout + "ms exceeded" : "timeout exceeded";
      const transitional2 = config2.transitional || transitional_default;
      if (config2.timeoutErrorMessage) {
        timeoutErrorMessage = config2.timeoutErrorMessage;
      }
      reject(new AxiosError_default(
        timeoutErrorMessage,
        transitional2.clarifyTimeoutError ? AxiosError_default.ETIMEDOUT : AxiosError_default.ECONNABORTED,
        config2,
        request2
      ));
      request2 = null;
    };
    if (browser_default.isStandardBrowserEnv) {
      const xsrfValue = (config2.withCredentials || isURLSameOrigin_default(fullPath)) && config2.xsrfCookieName && cookies_default.read(config2.xsrfCookieName);
      if (xsrfValue) {
        requestHeaders.set(config2.xsrfHeaderName, xsrfValue);
      }
    }
    requestData === void 0 && requestHeaders.setContentType(null);
    if ("setRequestHeader" in request2) {
      utils_default.forEach(requestHeaders.toJSON(), function setRequestHeader(val, key) {
        request2.setRequestHeader(key, val);
      });
    }
    if (!utils_default.isUndefined(config2.withCredentials)) {
      request2.withCredentials = !!config2.withCredentials;
    }
    if (responseType && responseType !== "json") {
      request2.responseType = config2.responseType;
    }
    if (typeof config2.onDownloadProgress === "function") {
      request2.addEventListener("progress", progressEventReducer(config2.onDownloadProgress, true));
    }
    if (typeof config2.onUploadProgress === "function" && request2.upload) {
      request2.upload.addEventListener("progress", progressEventReducer(config2.onUploadProgress));
    }
    if (config2.cancelToken || config2.signal) {
      onCanceled = (cancel) => {
        if (!request2) {
          return;
        }
        reject(!cancel || cancel.type ? new CanceledError_default(null, config2, request2) : cancel);
        request2.abort();
        request2 = null;
      };
      config2.cancelToken && config2.cancelToken.subscribe(onCanceled);
      if (config2.signal) {
        config2.signal.aborted ? onCanceled() : config2.signal.addEventListener("abort", onCanceled);
      }
    }
    const protocol = parseProtocol(fullPath);
    if (protocol && browser_default.protocols.indexOf(protocol) === -1) {
      reject(new AxiosError_default("Unsupported protocol " + protocol + ":", AxiosError_default.ERR_BAD_REQUEST, config2));
      return;
    }
    request2.send(requestData || null);
  });
};

// node_modules/axios/lib/adapters/adapters.js
var knownAdapters = {
  http: null_default,
  xhr: xhr_default
};
utils_default.forEach(knownAdapters, (fn, value) => {
  if (fn) {
    try {
      Object.defineProperty(fn, "name", { value });
    } catch (e10) {
    }
    Object.defineProperty(fn, "adapterName", { value });
  }
});
var adapters_default = {
  getAdapter: (adapters) => {
    adapters = utils_default.isArray(adapters) ? adapters : [adapters];
    const { length } = adapters;
    let nameOrAdapter;
    let adapter;
    for (let i7 = 0; i7 < length; i7++) {
      nameOrAdapter = adapters[i7];
      if (adapter = utils_default.isString(nameOrAdapter) ? knownAdapters[nameOrAdapter.toLowerCase()] : nameOrAdapter) {
        break;
      }
    }
    if (!adapter) {
      if (adapter === false) {
        throw new AxiosError_default(
          `Adapter ${nameOrAdapter} is not supported by the environment`,
          "ERR_NOT_SUPPORT"
        );
      }
      throw new Error(
        utils_default.hasOwnProp(knownAdapters, nameOrAdapter) ? `Adapter '${nameOrAdapter}' is not available in the build` : `Unknown adapter '${nameOrAdapter}'`
      );
    }
    if (!utils_default.isFunction(adapter)) {
      throw new TypeError("adapter is not a function");
    }
    return adapter;
  },
  adapters: knownAdapters
};

// node_modules/axios/lib/core/dispatchRequest.js
function throwIfCancellationRequested(config2) {
  if (config2.cancelToken) {
    config2.cancelToken.throwIfRequested();
  }
  if (config2.signal && config2.signal.aborted) {
    throw new CanceledError_default(null, config2);
  }
}
function dispatchRequest(config2) {
  throwIfCancellationRequested(config2);
  config2.headers = AxiosHeaders_default.from(config2.headers);
  config2.data = transformData.call(
    config2,
    config2.transformRequest
  );
  if (["post", "put", "patch"].indexOf(config2.method) !== -1) {
    config2.headers.setContentType("application/x-www-form-urlencoded", false);
  }
  const adapter = adapters_default.getAdapter(config2.adapter || defaults_default.adapter);
  return adapter(config2).then(function onAdapterResolution(response) {
    throwIfCancellationRequested(config2);
    response.data = transformData.call(
      config2,
      config2.transformResponse,
      response
    );
    response.headers = AxiosHeaders_default.from(response.headers);
    return response;
  }, function onAdapterRejection(reason) {
    if (!isCancel(reason)) {
      throwIfCancellationRequested(config2);
      if (reason && reason.response) {
        reason.response.data = transformData.call(
          config2,
          config2.transformResponse,
          reason.response
        );
        reason.response.headers = AxiosHeaders_default.from(reason.response.headers);
      }
    }
    return Promise.reject(reason);
  });
}

// node_modules/axios/lib/core/mergeConfig.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var headersToObject = (thing) => thing instanceof AxiosHeaders_default ? thing.toJSON() : thing;
function mergeConfig(config1, config2) {
  config2 = config2 || {};
  const config3 = {};
  function getMergedValue(target, source, caseless) {
    if (utils_default.isPlainObject(target) && utils_default.isPlainObject(source)) {
      return utils_default.merge.call({ caseless }, target, source);
    } else if (utils_default.isPlainObject(source)) {
      return utils_default.merge({}, source);
    } else if (utils_default.isArray(source)) {
      return source.slice();
    }
    return source;
  }
  function mergeDeepProperties(a7, b5, caseless) {
    if (!utils_default.isUndefined(b5)) {
      return getMergedValue(a7, b5, caseless);
    } else if (!utils_default.isUndefined(a7)) {
      return getMergedValue(void 0, a7, caseless);
    }
  }
  function valueFromConfig2(a7, b5) {
    if (!utils_default.isUndefined(b5)) {
      return getMergedValue(void 0, b5);
    }
  }
  function defaultToConfig2(a7, b5) {
    if (!utils_default.isUndefined(b5)) {
      return getMergedValue(void 0, b5);
    } else if (!utils_default.isUndefined(a7)) {
      return getMergedValue(void 0, a7);
    }
  }
  function mergeDirectKeys(a7, b5, prop) {
    if (prop in config2) {
      return getMergedValue(a7, b5);
    } else if (prop in config1) {
      return getMergedValue(void 0, a7);
    }
  }
  const mergeMap = {
    url: valueFromConfig2,
    method: valueFromConfig2,
    data: valueFromConfig2,
    baseURL: defaultToConfig2,
    transformRequest: defaultToConfig2,
    transformResponse: defaultToConfig2,
    paramsSerializer: defaultToConfig2,
    timeout: defaultToConfig2,
    timeoutMessage: defaultToConfig2,
    withCredentials: defaultToConfig2,
    adapter: defaultToConfig2,
    responseType: defaultToConfig2,
    xsrfCookieName: defaultToConfig2,
    xsrfHeaderName: defaultToConfig2,
    onUploadProgress: defaultToConfig2,
    onDownloadProgress: defaultToConfig2,
    decompress: defaultToConfig2,
    maxContentLength: defaultToConfig2,
    maxBodyLength: defaultToConfig2,
    beforeRedirect: defaultToConfig2,
    transport: defaultToConfig2,
    httpAgent: defaultToConfig2,
    httpsAgent: defaultToConfig2,
    cancelToken: defaultToConfig2,
    socketPath: defaultToConfig2,
    responseEncoding: defaultToConfig2,
    validateStatus: mergeDirectKeys,
    headers: (a7, b5) => mergeDeepProperties(headersToObject(a7), headersToObject(b5), true)
  };
  utils_default.forEach(Object.keys(Object.assign({}, config1, config2)), function computeConfigValue(prop) {
    const merge2 = mergeMap[prop] || mergeDeepProperties;
    const configValue = merge2(config1[prop], config2[prop], prop);
    utils_default.isUndefined(configValue) && merge2 !== mergeDirectKeys || (config3[prop] = configValue);
  });
  return config3;
}

// node_modules/axios/lib/helpers/validator.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/axios/lib/env/data.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var VERSION = "1.4.0";

// node_modules/axios/lib/helpers/validator.js
var validators = {};
["object", "boolean", "number", "function", "string", "symbol"].forEach((type2, i7) => {
  validators[type2] = function validator(thing) {
    return typeof thing === type2 || "a" + (i7 < 1 ? "n " : " ") + type2;
  };
});
var deprecatedWarnings = {};
validators.transitional = function transitional(validator, version4, message) {
  function formatMessage(opt, desc) {
    return "[Axios v" + VERSION + "] Transitional option '" + opt + "'" + desc + (message ? ". " + message : "");
  }
  return (value, opt, opts) => {
    if (validator === false) {
      throw new AxiosError_default(
        formatMessage(opt, " has been removed" + (version4 ? " in " + version4 : "")),
        AxiosError_default.ERR_DEPRECATED
      );
    }
    if (version4 && !deprecatedWarnings[opt]) {
      deprecatedWarnings[opt] = true;
      console.warn(
        formatMessage(
          opt,
          " has been deprecated since v" + version4 + " and will be removed in the near future"
        )
      );
    }
    return validator ? validator(value, opt, opts) : true;
  };
};
function assertOptions(options, schema, allowUnknown) {
  if (typeof options !== "object") {
    throw new AxiosError_default("options must be an object", AxiosError_default.ERR_BAD_OPTION_VALUE);
  }
  const keys = Object.keys(options);
  let i7 = keys.length;
  while (i7-- > 0) {
    const opt = keys[i7];
    const validator = schema[opt];
    if (validator) {
      const value = options[opt];
      const result = value === void 0 || validator(value, opt, options);
      if (result !== true) {
        throw new AxiosError_default("option " + opt + " must be " + result, AxiosError_default.ERR_BAD_OPTION_VALUE);
      }
      continue;
    }
    if (allowUnknown !== true) {
      throw new AxiosError_default("Unknown option " + opt, AxiosError_default.ERR_BAD_OPTION);
    }
  }
}
var validator_default = {
  assertOptions,
  validators
};

// node_modules/axios/lib/core/Axios.js
var validators2 = validator_default.validators;
var Axios = class {
  constructor(instanceConfig) {
    this.defaults = instanceConfig;
    this.interceptors = {
      request: new InterceptorManager_default(),
      response: new InterceptorManager_default()
    };
  }
  /**
   * Dispatch a request
   *
   * @param {String|Object} configOrUrl The config specific for this request (merged with this.defaults)
   * @param {?Object} config
   *
   * @returns {Promise} The Promise to be fulfilled
   */
  request(configOrUrl, config2) {
    if (typeof configOrUrl === "string") {
      config2 = config2 || {};
      config2.url = configOrUrl;
    } else {
      config2 = configOrUrl || {};
    }
    config2 = mergeConfig(this.defaults, config2);
    const { transitional: transitional2, paramsSerializer, headers } = config2;
    if (transitional2 !== void 0) {
      validator_default.assertOptions(transitional2, {
        silentJSONParsing: validators2.transitional(validators2.boolean),
        forcedJSONParsing: validators2.transitional(validators2.boolean),
        clarifyTimeoutError: validators2.transitional(validators2.boolean)
      }, false);
    }
    if (paramsSerializer != null) {
      if (utils_default.isFunction(paramsSerializer)) {
        config2.paramsSerializer = {
          serialize: paramsSerializer
        };
      } else {
        validator_default.assertOptions(paramsSerializer, {
          encode: validators2.function,
          serialize: validators2.function
        }, true);
      }
    }
    config2.method = (config2.method || this.defaults.method || "get").toLowerCase();
    let contextHeaders;
    contextHeaders = headers && utils_default.merge(
      headers.common,
      headers[config2.method]
    );
    contextHeaders && utils_default.forEach(
      ["delete", "get", "head", "post", "put", "patch", "common"],
      (method) => {
        delete headers[method];
      }
    );
    config2.headers = AxiosHeaders_default.concat(contextHeaders, headers);
    const requestInterceptorChain = [];
    let synchronousRequestInterceptors = true;
    this.interceptors.request.forEach(function unshiftRequestInterceptors(interceptor) {
      if (typeof interceptor.runWhen === "function" && interceptor.runWhen(config2) === false) {
        return;
      }
      synchronousRequestInterceptors = synchronousRequestInterceptors && interceptor.synchronous;
      requestInterceptorChain.unshift(interceptor.fulfilled, interceptor.rejected);
    });
    const responseInterceptorChain = [];
    this.interceptors.response.forEach(function pushResponseInterceptors(interceptor) {
      responseInterceptorChain.push(interceptor.fulfilled, interceptor.rejected);
    });
    let promise;
    let i7 = 0;
    let len;
    if (!synchronousRequestInterceptors) {
      const chain = [dispatchRequest.bind(this), void 0];
      chain.unshift.apply(chain, requestInterceptorChain);
      chain.push.apply(chain, responseInterceptorChain);
      len = chain.length;
      promise = Promise.resolve(config2);
      while (i7 < len) {
        promise = promise.then(chain[i7++], chain[i7++]);
      }
      return promise;
    }
    len = requestInterceptorChain.length;
    let newConfig = config2;
    i7 = 0;
    while (i7 < len) {
      const onFulfilled = requestInterceptorChain[i7++];
      const onRejected = requestInterceptorChain[i7++];
      try {
        newConfig = onFulfilled(newConfig);
      } catch (error) {
        onRejected.call(this, error);
        break;
      }
    }
    try {
      promise = dispatchRequest.call(this, newConfig);
    } catch (error) {
      return Promise.reject(error);
    }
    i7 = 0;
    len = responseInterceptorChain.length;
    while (i7 < len) {
      promise = promise.then(responseInterceptorChain[i7++], responseInterceptorChain[i7++]);
    }
    return promise;
  }
  getUri(config2) {
    config2 = mergeConfig(this.defaults, config2);
    const fullPath = buildFullPath(config2.baseURL, config2.url);
    return buildURL(fullPath, config2.params, config2.paramsSerializer);
  }
};
utils_default.forEach(["delete", "get", "head", "options"], function forEachMethodNoData2(method) {
  Axios.prototype[method] = function(url, config2) {
    return this.request(mergeConfig(config2 || {}, {
      method,
      url,
      data: (config2 || {}).data
    }));
  };
});
utils_default.forEach(["post", "put", "patch"], function forEachMethodWithData2(method) {
  function generateHTTPMethod(isForm) {
    return function httpMethod(url, data, config2) {
      return this.request(mergeConfig(config2 || {}, {
        method,
        headers: isForm ? {
          "Content-Type": "multipart/form-data"
        } : {},
        url,
        data
      }));
    };
  }
  Axios.prototype[method] = generateHTTPMethod();
  Axios.prototype[method + "Form"] = generateHTTPMethod(true);
});
var Axios_default = Axios;

// node_modules/axios/lib/cancel/CancelToken.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var CancelToken = class {
  constructor(executor) {
    if (typeof executor !== "function") {
      throw new TypeError("executor must be a function.");
    }
    let resolvePromise;
    this.promise = new Promise(function promiseExecutor(resolve4) {
      resolvePromise = resolve4;
    });
    const token = this;
    this.promise.then((cancel) => {
      if (!token._listeners)
        return;
      let i7 = token._listeners.length;
      while (i7-- > 0) {
        token._listeners[i7](cancel);
      }
      token._listeners = null;
    });
    this.promise.then = (onfulfilled) => {
      let _resolve2;
      const promise = new Promise((resolve4) => {
        token.subscribe(resolve4);
        _resolve2 = resolve4;
      }).then(onfulfilled);
      promise.cancel = function reject() {
        token.unsubscribe(_resolve2);
      };
      return promise;
    };
    executor(function cancel(message, config2, request2) {
      if (token.reason) {
        return;
      }
      token.reason = new CanceledError_default(message, config2, request2);
      resolvePromise(token.reason);
    });
  }
  /**
   * Throws a `CanceledError` if cancellation has been requested.
   */
  throwIfRequested() {
    if (this.reason) {
      throw this.reason;
    }
  }
  /**
   * Subscribe to the cancel signal
   */
  subscribe(listener) {
    if (this.reason) {
      listener(this.reason);
      return;
    }
    if (this._listeners) {
      this._listeners.push(listener);
    } else {
      this._listeners = [listener];
    }
  }
  /**
   * Unsubscribe from the cancel signal
   */
  unsubscribe(listener) {
    if (!this._listeners) {
      return;
    }
    const index = this._listeners.indexOf(listener);
    if (index !== -1) {
      this._listeners.splice(index, 1);
    }
  }
  /**
   * Returns an object that contains a new `CancelToken` and a function that, when called,
   * cancels the `CancelToken`.
   */
  static source() {
    let cancel;
    const token = new CancelToken(function executor(c7) {
      cancel = c7;
    });
    return {
      token,
      cancel
    };
  }
};
var CancelToken_default = CancelToken;

// node_modules/axios/lib/helpers/spread.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function spread(callback) {
  return function wrap(arr) {
    return callback.apply(null, arr);
  };
}

// node_modules/axios/lib/helpers/isAxiosError.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function isAxiosError(payload) {
  return utils_default.isObject(payload) && payload.isAxiosError === true;
}

// node_modules/axios/lib/helpers/HttpStatusCode.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var HttpStatusCode = {
  Continue: 100,
  SwitchingProtocols: 101,
  Processing: 102,
  EarlyHints: 103,
  Ok: 200,
  Created: 201,
  Accepted: 202,
  NonAuthoritativeInformation: 203,
  NoContent: 204,
  ResetContent: 205,
  PartialContent: 206,
  MultiStatus: 207,
  AlreadyReported: 208,
  ImUsed: 226,
  MultipleChoices: 300,
  MovedPermanently: 301,
  Found: 302,
  SeeOther: 303,
  NotModified: 304,
  UseProxy: 305,
  Unused: 306,
  TemporaryRedirect: 307,
  PermanentRedirect: 308,
  BadRequest: 400,
  Unauthorized: 401,
  PaymentRequired: 402,
  Forbidden: 403,
  NotFound: 404,
  MethodNotAllowed: 405,
  NotAcceptable: 406,
  ProxyAuthenticationRequired: 407,
  RequestTimeout: 408,
  Conflict: 409,
  Gone: 410,
  LengthRequired: 411,
  PreconditionFailed: 412,
  PayloadTooLarge: 413,
  UriTooLong: 414,
  UnsupportedMediaType: 415,
  RangeNotSatisfiable: 416,
  ExpectationFailed: 417,
  ImATeapot: 418,
  MisdirectedRequest: 421,
  UnprocessableEntity: 422,
  Locked: 423,
  FailedDependency: 424,
  TooEarly: 425,
  UpgradeRequired: 426,
  PreconditionRequired: 428,
  TooManyRequests: 429,
  RequestHeaderFieldsTooLarge: 431,
  UnavailableForLegalReasons: 451,
  InternalServerError: 500,
  NotImplemented: 501,
  BadGateway: 502,
  ServiceUnavailable: 503,
  GatewayTimeout: 504,
  HttpVersionNotSupported: 505,
  VariantAlsoNegotiates: 506,
  InsufficientStorage: 507,
  LoopDetected: 508,
  NotExtended: 510,
  NetworkAuthenticationRequired: 511
};
Object.entries(HttpStatusCode).forEach(([key, value]) => {
  HttpStatusCode[value] = key;
});
var HttpStatusCode_default = HttpStatusCode;

// node_modules/axios/lib/axios.js
function createInstance(defaultConfig) {
  const context = new Axios_default(defaultConfig);
  const instance = bind(Axios_default.prototype.request, context);
  utils_default.extend(instance, Axios_default.prototype, context, { allOwnKeys: true });
  utils_default.extend(instance, context, null, { allOwnKeys: true });
  instance.create = function create(instanceConfig) {
    return createInstance(mergeConfig(defaultConfig, instanceConfig));
  };
  return instance;
}
var axios = createInstance(defaults_default);
axios.Axios = Axios_default;
axios.CanceledError = CanceledError_default;
axios.CancelToken = CancelToken_default;
axios.isCancel = isCancel;
axios.VERSION = VERSION;
axios.toFormData = toFormData_default;
axios.AxiosError = AxiosError_default;
axios.Cancel = axios.CanceledError;
axios.all = function all(promises3) {
  return Promise.all(promises3);
};
axios.spread = spread;
axios.isAxiosError = isAxiosError;
axios.mergeConfig = mergeConfig;
axios.AxiosHeaders = AxiosHeaders_default;
axios.formToJSON = (thing) => formDataToJSON_default(utils_default.isHTMLForm(thing) ? new FormData(thing) : thing);
axios.HttpStatusCode = HttpStatusCode_default;
axios.default = axios;
var axios_default = axios;

// src/generated/core/request.ts
var import_form_data = __toESM(require_browser());

// src/generated/core/ApiError.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var ApiError = class extends Error {
  constructor(request2, response, message) {
    super(message);
    this.name = "ApiError";
    this.url = response.url;
    this.status = response.status;
    this.statusText = response.statusText;
    this.body = response.body;
    this.request = request2;
  }
};

// src/generated/core/CancelablePromise.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var CancelError = class extends Error {
  constructor(message) {
    super(message);
    this.name = "CancelError";
  }
  get isCancelled() {
    return true;
  }
};
var _isResolved, _isRejected, _isCancelled, _cancelHandlers, _promise, _resolve, _reject;
var CancelablePromise = class {
  constructor(executor) {
    __privateAdd(this, _isResolved, void 0);
    __privateAdd(this, _isRejected, void 0);
    __privateAdd(this, _isCancelled, void 0);
    __privateAdd(this, _cancelHandlers, void 0);
    __privateAdd(this, _promise, void 0);
    __privateAdd(this, _resolve, void 0);
    __privateAdd(this, _reject, void 0);
    __privateSet(this, _isResolved, false);
    __privateSet(this, _isRejected, false);
    __privateSet(this, _isCancelled, false);
    __privateSet(this, _cancelHandlers, []);
    __privateSet(this, _promise, new Promise((resolve4, reject) => {
      __privateSet(this, _resolve, resolve4);
      __privateSet(this, _reject, reject);
      const onResolve = (value) => {
        var _a;
        if (__privateGet(this, _isResolved) || __privateGet(this, _isRejected) || __privateGet(this, _isCancelled)) {
          return;
        }
        __privateSet(this, _isResolved, true);
        (_a = __privateGet(this, _resolve)) == null ? void 0 : _a.call(this, value);
      };
      const onReject = (reason) => {
        var _a;
        if (__privateGet(this, _isResolved) || __privateGet(this, _isRejected) || __privateGet(this, _isCancelled)) {
          return;
        }
        __privateSet(this, _isRejected, true);
        (_a = __privateGet(this, _reject)) == null ? void 0 : _a.call(this, reason);
      };
      const onCancel = (cancelHandler) => {
        if (__privateGet(this, _isResolved) || __privateGet(this, _isRejected) || __privateGet(this, _isCancelled)) {
          return;
        }
        __privateGet(this, _cancelHandlers).push(cancelHandler);
      };
      Object.defineProperty(onCancel, "isResolved", {
        get: () => __privateGet(this, _isResolved)
      });
      Object.defineProperty(onCancel, "isRejected", {
        get: () => __privateGet(this, _isRejected)
      });
      Object.defineProperty(onCancel, "isCancelled", {
        get: () => __privateGet(this, _isCancelled)
      });
      return executor(onResolve, onReject, onCancel);
    }));
  }
  get [Symbol.toStringTag]() {
    return "Cancellable Promise";
  }
  then(onFulfilled, onRejected) {
    return __privateGet(this, _promise).then(onFulfilled, onRejected);
  }
  catch(onRejected) {
    return __privateGet(this, _promise).catch(onRejected);
  }
  finally(onFinally) {
    return __privateGet(this, _promise).finally(onFinally);
  }
  cancel() {
    var _a;
    if (__privateGet(this, _isResolved) || __privateGet(this, _isRejected) || __privateGet(this, _isCancelled)) {
      return;
    }
    __privateSet(this, _isCancelled, true);
    if (__privateGet(this, _cancelHandlers).length) {
      try {
        for (const cancelHandler of __privateGet(this, _cancelHandlers)) {
          cancelHandler();
        }
      } catch (error) {
        console.warn("Cancellation threw an error", error);
        return;
      }
    }
    __privateGet(this, _cancelHandlers).length = 0;
    (_a = __privateGet(this, _reject)) == null ? void 0 : _a.call(this, new CancelError("Request aborted"));
  }
  get isCancelled() {
    return __privateGet(this, _isCancelled);
  }
};
_isResolved = new WeakMap();
_isRejected = new WeakMap();
_isCancelled = new WeakMap();
_cancelHandlers = new WeakMap();
_promise = new WeakMap();
_resolve = new WeakMap();
_reject = new WeakMap();

// src/generated/core/request.ts
var isDefined = (value) => {
  return value !== void 0 && value !== null;
};
var isString2 = (value) => {
  return typeof value === "string";
};
var isStringWithValue = (value) => {
  return isString2(value) && value !== "";
};
var isBlob2 = (value) => {
  return typeof value === "object" && typeof value.type === "string" && typeof value.stream === "function" && typeof value.arrayBuffer === "function" && typeof value.constructor === "function" && typeof value.constructor.name === "string" && /^(Blob|File)$/.test(value.constructor.name) && /^(Blob|File)$/.test(value[Symbol.toStringTag]);
};
var isFormData2 = (value) => {
  return value instanceof import_form_data.default;
};
var isSuccess = (status) => {
  return status >= 200 && status < 300;
};
var base64 = (str) => {
  try {
    return btoa(str);
  } catch (err) {
    return Buffer2.from(str).toString("base64");
  }
};
var getQueryString = (params) => {
  const qs = [];
  const append2 = (key, value) => {
    qs.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
  };
  const process3 = (key, value) => {
    if (isDefined(value)) {
      if (Array.isArray(value)) {
        value.forEach((v7) => {
          process3(key, v7);
        });
      } else if (typeof value === "object") {
        Object.entries(value).forEach(([k4, v7]) => {
          process3(`${key}[${k4}]`, v7);
        });
      } else {
        append2(key, value);
      }
    }
  };
  Object.entries(params).forEach(([key, value]) => {
    process3(key, value);
  });
  if (qs.length > 0) {
    return `?${qs.join("&")}`;
  }
  return "";
};
var getUrl = (config2, options) => {
  const encoder = config2.ENCODE_PATH || encodeURI;
  const path2 = options.url.replace("{api-version}", config2.VERSION).replace(/{(.*?)}/g, (substring, group) => {
    if (options.path?.hasOwnProperty(group)) {
      return encoder(String(options.path[group]));
    }
    return substring;
  });
  const url = `${config2.BASE}${path2}`;
  if (options.query) {
    return `${url}${getQueryString(options.query)}`;
  }
  return url;
};
var getFormData = (options) => {
  if (options.formData) {
    const formData = new import_form_data.default();
    const process3 = (key, value) => {
      if (isString2(value) || isBlob2(value)) {
        formData.append(key, value);
      } else {
        formData.append(key, JSON.stringify(value));
      }
    };
    Object.entries(options.formData).filter(([_4, value]) => isDefined(value)).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((v7) => process3(key, v7));
      } else {
        process3(key, value);
      }
    });
    return formData;
  }
  return void 0;
};
var resolve = async (options, resolver) => {
  if (typeof resolver === "function") {
    return resolver(options);
  }
  return resolver;
};
var getHeaders = async (config2, options, formData) => {
  const token = await resolve(options, config2.TOKEN);
  const username = await resolve(options, config2.USERNAME);
  const password = await resolve(options, config2.PASSWORD);
  const additionalHeaders = await resolve(options, config2.HEADERS);
  const formHeaders = typeof formData?.getHeaders === "function" && formData?.getHeaders() || {};
  const headers = Object.entries({
    Accept: "application/json",
    ...additionalHeaders,
    ...options.headers,
    ...formHeaders
  }).filter(([_4, value]) => isDefined(value)).reduce((headers2, [key, value]) => ({
    ...headers2,
    [key]: String(value)
  }), {});
  if (isStringWithValue(token)) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (isStringWithValue(username) && isStringWithValue(password)) {
    const credentials = base64(`${username}:${password}`);
    headers["Authorization"] = `Basic ${credentials}`;
  }
  if (options.body) {
    if (options.mediaType) {
      headers["Content-Type"] = options.mediaType;
    } else if (isBlob2(options.body)) {
      headers["Content-Type"] = options.body.type || "application/octet-stream";
    } else if (isString2(options.body)) {
      headers["Content-Type"] = "text/plain";
    } else if (!isFormData2(options.body)) {
      headers["Content-Type"] = "application/json";
    }
  }
  return headers;
};
var getRequestBody = (options) => {
  if (options.body) {
    return options.body;
  }
  return void 0;
};
var sendRequest = async (config2, options, url, body, formData, headers, onCancel) => {
  const source = axios_default.CancelToken.source();
  const requestConfig = {
    url,
    headers,
    data: body ?? formData,
    method: options.method,
    withCredentials: config2.WITH_CREDENTIALS,
    cancelToken: source.token
  };
  onCancel(() => source.cancel("The user aborted a request."));
  try {
    return await axios_default.request(requestConfig);
  } catch (error) {
    const axiosError = error;
    if (axiosError.response) {
      return axiosError.response;
    }
    throw error;
  }
};
var getResponseHeader = (response, responseHeader) => {
  if (responseHeader) {
    const content = response.headers[responseHeader];
    if (isString2(content)) {
      return content;
    }
  }
  return void 0;
};
var getResponseBody = (response) => {
  if (response.status !== 204) {
    return response.data;
  }
  return void 0;
};
var catchErrorCodes = (options, result) => {
  const errors = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    ...options.errors
  };
  const error = errors[result.status];
  if (error) {
    throw new ApiError(options, result, error);
  }
  if (!result.ok) {
    throw new ApiError(options, result, "Generic Error");
  }
};
var request = (config2, options) => {
  return new CancelablePromise(async (resolve4, reject, onCancel) => {
    try {
      const url = getUrl(config2, options);
      const formData = getFormData(options);
      const body = getRequestBody(options);
      const headers = await getHeaders(config2, options, formData);
      if (!onCancel.isCancelled) {
        const response = await sendRequest(config2, options, url, body, formData, headers, onCancel);
        const responseBody = getResponseBody(response);
        const responseHeader = getResponseHeader(response, options.responseHeader);
        const result = {
          url,
          ok: isSuccess(response.status),
          status: response.status,
          statusText: response.statusText,
          body: responseHeader ?? responseBody
        };
        catchErrorCodes(options, result);
        resolve4(result.body);
      }
    } catch (error) {
      reject(error);
    }
  });
};

// src/generated/core/AxiosHttpRequest.ts
var AxiosHttpRequest = class extends BaseHttpRequest {
  constructor(config2) {
    super(config2);
  }
  /**
   * Request method
   * @param options The request options from the service
   * @returns CancelablePromise<T>
   * @throws ApiError
   */
  request(options) {
    return request(this.config, options);
  }
};

// src/generated/services/V1Service.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var V1Service = class {
  constructor(httpRequest) {
    this.httpRequest = httpRequest;
  }
  /**
   * @param requestBody
   * @returns CompletionResponse Success
   * @throws ApiError
   */
  completion(requestBody) {
    return this.httpRequest.request({
      method: "POST",
      url: "/v1/completions",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        400: `Bad Request`
      }
    });
  }
  /**
   * @param requestBody
   * @returns any Success
   * @throws ApiError
   */
  event(requestBody) {
    return this.httpRequest.request({
      method: "POST",
      url: "/v1/events",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        400: `Bad Request`
      }
    });
  }
  /**
   * @returns HealthState Success
   * @throws ApiError
   */
  health() {
    return this.httpRequest.request({
      method: "POST",
      url: "/v1/health"
    });
  }
};

// src/generated/TabbyApi.ts
var TabbyApi = class {
  constructor(config2, HttpRequest = AxiosHttpRequest) {
    this.request = new HttpRequest({
      BASE: config2?.BASE ?? "https://playground.app.tabbyml.com/tabby",
      VERSION: config2?.VERSION ?? "0.1.0",
      WITH_CREDENTIALS: config2?.WITH_CREDENTIALS ?? false,
      CREDENTIALS: config2?.CREDENTIALS ?? "include",
      TOKEN: config2?.TOKEN,
      USERNAME: config2?.USERNAME,
      PASSWORD: config2?.PASSWORD,
      HEADERS: config2?.HEADERS,
      ENCODE_PATH: config2?.ENCODE_PATH
    });
    this.v1 = new V1Service(this.request);
  }
};

// src/generated/core/OpenAPI.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/utils.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function splitLines(input) {
  return input.match(/.*(?:$|\r?\n)/g).filter(Boolean);
}
function splitWords(input) {
  return input.match(/\w+|\W+/g).filter(Boolean);
}
function isBlank(input) {
  return input.trim().length === 0;
}
function cancelable(promise, cancel) {
  return new CancelablePromise((resolve4, reject, onCancel) => {
    promise.then((resp) => {
      resolve4(resp);
    }).catch((err) => {
      reject(err);
    });
    onCancel(() => {
      cancel();
    });
  });
}

// src/Auth.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
init_events();

// node_modules/jwt-decode/build/jwt-decode.esm.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function e2(e10) {
  this.message = e10;
}
e2.prototype = new Error(), e2.prototype.name = "InvalidCharacterError";
var r2 = "undefined" != typeof window && window.atob && window.atob.bind(window) || function(r10) {
  var t9 = String(r10).replace(/=+$/, "");
  if (t9.length % 4 == 1)
    throw new e2("'atob' failed: The string to be decoded is not correctly encoded.");
  for (var n9, o9, a7 = 0, i7 = 0, c7 = ""; o9 = t9.charAt(i7++); ~o9 && (n9 = a7 % 4 ? 64 * n9 + o9 : o9, a7++ % 4) ? c7 += String.fromCharCode(255 & n9 >> (-2 * a7 & 6)) : 0)
    o9 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(o9);
  return c7;
};
function t2(e10) {
  var t9 = e10.replace(/-/g, "+").replace(/_/g, "/");
  switch (t9.length % 4) {
    case 0:
      break;
    case 2:
      t9 += "==";
      break;
    case 3:
      t9 += "=";
      break;
    default:
      throw "Illegal base64url string!";
  }
  try {
    return function(e11) {
      return decodeURIComponent(r2(e11).replace(/(.)/g, function(e12, r10) {
        var t10 = r10.charCodeAt(0).toString(16).toUpperCase();
        return t10.length < 2 && (t10 = "0" + t10), "%" + t10;
      }));
    }(t9);
  } catch (e11) {
    return r2(t9);
  }
}
function n2(e10) {
  this.message = e10;
}
function o2(e10, r10) {
  if ("string" != typeof e10)
    throw new n2("Invalid token specified");
  var o9 = true === (r10 = r10 || {}).header ? 0 : 1;
  try {
    return JSON.parse(t2(e10.split(".")[o9]));
  } catch (e11) {
    throw new n2("Invalid token specified: " + e11.message);
  }
}
n2.prototype = new Error(), n2.prototype.name = "InvalidTokenError";
var jwt_decode_esm_default = o2;

// src/cloud/index.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/cloud/CloudApi.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/cloud/services/ApiService.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var ApiService = class {
  constructor(httpRequest) {
    this.httpRequest = httpRequest;
  }
  /**
   * @returns DeviceTokenResponse Success
   * @throws ApiError
   */
  deviceToken(body) {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token",
      body
    });
  }
  /**
   * @param code
   * @returns DeviceTokenAcceptResponse Success
   * @throws ApiError
   */
  deviceTokenAccept(query) {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token/accept",
      query
    });
  }
  /**
   * @param token
   * @returns DeviceTokenRefreshResponse Success
   * @throws ApiError
   */
  deviceTokenRefresh(token) {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token/refresh",
      headers: { Authorization: `Bearer ${token}` }
    });
  }
  /**
   * @param body object for anonymous usage tracking
   */
  usage(body) {
    return this.httpRequest.request({
      method: "POST",
      url: "/usage",
      body
    });
  }
};

// src/cloud/CloudApi.ts
var CloudApi = class {
  constructor(config2, HttpRequest = AxiosHttpRequest) {
    this.request = new HttpRequest({
      BASE: config2?.BASE ?? "https://app.tabbyml.com/api",
      VERSION: config2?.VERSION ?? "0.0.0",
      WITH_CREDENTIALS: config2?.WITH_CREDENTIALS ?? false,
      CREDENTIALS: config2?.CREDENTIALS ?? "include",
      TOKEN: config2?.TOKEN,
      USERNAME: config2?.USERNAME,
      PASSWORD: config2?.PASSWORD,
      HEADERS: config2?.HEADERS,
      ENCODE_PATH: config2?.ENCODE_PATH
    });
    this.api = new ApiService(this.request);
  }
};

// src/cloud/models/DeviceTokenResponse.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/cloud/models/DeviceTokenAcceptResponse.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/dataStore.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/env.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/dataStore.ts
var dataStore = null ;

// src/logger.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var import_pino = __toESM(require_browser2());
var rootLogger = (0, import_pino.default)();
var allLoggers = [rootLogger];
rootLogger.onChild = (child) => {
  allLoggers.push(child);
};

// src/Auth.ts
var _Auth = class extends EventEmitter {
  constructor(options) {
    super();
    this.logger = rootLogger.child({ component: "Auth" });
    this.dataStore = null;
    this.refreshTokenTimer = null;
    this.authApi = null;
    this.jwt = null;
    this.endpoint = options.endpoint;
    this.dataStore = options.dataStore || dataStore;
    this.authApi = new CloudApi();
    this.scheduleRefreshToken();
  }
  static async create(options) {
    const auth = new _Auth(options);
    await auth.load();
    return auth;
  }
  get token() {
    return this.jwt?.token;
  }
  get user() {
    return this.jwt?.payload.email;
  }
  async load() {
    if (!this.dataStore)
      return;
    try {
      await this.dataStore.load();
      const storedJwt = this.dataStore.data["auth"]?.[this.endpoint]?.jwt;
      if (typeof storedJwt === "string" && this.jwt?.token !== storedJwt) {
        this.logger.debug({ storedJwt }, "Load jwt from data store.");
        const jwt = {
          token: storedJwt,
          payload: jwt_decode_esm_default(storedJwt)
        };
        if (jwt.payload.exp * 1e3 - Date.now() < _Auth.tokenStrategy.refresh.beforeExpire) {
          this.jwt = await this.refreshToken(jwt, _Auth.tokenStrategy.refresh.whenLoaded);
          await this.save();
        } else {
          this.jwt = jwt;
        }
      }
    } catch (error) {
      this.logger.debug({ error }, "Error when loading auth");
    }
  }
  async save() {
    if (!this.dataStore)
      return;
    try {
      if (this.jwt) {
        if (this.dataStore.data["auth"]?.[this.endpoint]?.jwt === this.jwt.token)
          return;
        this.dataStore.data["auth"] = { ...this.dataStore.data["auth"], [this.endpoint]: { jwt: this.jwt.token } };
      } else {
        if (typeof this.dataStore.data["auth"]?.[this.endpoint] === "undefined")
          return;
        delete this.dataStore.data["auth"][this.endpoint];
      }
      await this.dataStore.save();
      this.logger.debug("Save changes to data store.");
    } catch (error) {
      this.logger.error({ error }, "Error when saving auth");
    }
  }
  async reset() {
    if (this.jwt) {
      this.jwt = null;
      await this.save();
    }
  }
  requestAuthUrl() {
    return new CancelablePromise(async (resolve4, reject, onCancel) => {
      let apiRequest;
      onCancel(() => {
        apiRequest?.cancel();
      });
      try {
        await this.reset();
        if (onCancel.isCancelled)
          return;
        this.logger.debug("Start to request device token");
        apiRequest = this.authApi.api.deviceToken({ auth_url: this.endpoint });
        const deviceToken = await apiRequest;
        this.logger.debug({ deviceToken }, "Request device token response");
        const authUrl = new URL(_Auth.authPageUrl);
        authUrl.searchParams.append("code", deviceToken.data.code);
        resolve4({ authUrl: authUrl.toString(), code: deviceToken.data.code });
      } catch (error) {
        this.logger.error({ error }, "Error when requesting token");
        reject(error);
      }
    });
  }
  pollingToken(code) {
    return new CancelablePromise((resolve4, reject, onCancel) => {
      let apiRequest;
      const timer = setInterval(async () => {
        try {
          apiRequest = this.authApi.api.deviceTokenAccept({ code });
          const response = await apiRequest;
          this.logger.debug({ response }, "Poll jwt response");
          this.jwt = {
            token: response.data.jwt,
            payload: jwt_decode_esm_default(response.data.jwt)
          };
          super.emit("updated", this.jwt);
          await this.save();
          clearInterval(timer);
          resolve4(true);
        } catch (error) {
          if (error instanceof ApiError && [400, 401, 403, 405].indexOf(error.status) !== -1) {
            this.logger.debug({ error }, "Expected error when polling jwt");
          } else {
            this.logger.error({ error }, "Error when polling jwt");
          }
        }
      }, _Auth.tokenStrategy.polling.interval);
      setTimeout(() => {
        clearInterval(timer);
        reject(new Error("Timeout when polling token"));
      }, _Auth.tokenStrategy.polling.timeout);
      onCancel(() => {
        apiRequest?.cancel();
        clearInterval(timer);
      });
    });
  }
  async refreshToken(jwt, options = { maxTry: 1, retryDelay: 1e3 }, retry = 0) {
    try {
      this.logger.debug({ retry }, "Start to refresh token");
      const refreshedJwt = await this.authApi.api.deviceTokenRefresh(jwt.token);
      this.logger.debug({ refreshedJwt }, "Refresh token response");
      return {
        token: refreshedJwt.data.jwt,
        payload: jwt_decode_esm_default(refreshedJwt.data.jwt)
      };
    } catch (error) {
      if (error instanceof ApiError && [400, 401, 403, 405].indexOf(error.status) !== -1) {
        this.logger.debug({ error }, "Error when refreshing jwt");
      } else {
        this.logger.error({ error }, "Unknown error when refreshing jwt");
        if (retry < options.maxTry) {
          this.logger.debug(`Retry refreshing jwt after ${options.retryDelay}ms`);
          await new Promise((resolve4) => setTimeout(resolve4, options.retryDelay));
          return this.refreshToken(jwt, options, retry + 1);
        }
      }
      throw { ...error, retry };
    }
  }
  scheduleRefreshToken() {
    this.refreshTokenTimer = setInterval(async () => {
      if (!this.jwt) {
        return null;
      }
      if (this.jwt.payload.exp * 1e3 - Date.now() < _Auth.tokenStrategy.refresh.beforeExpire) {
        try {
          this.jwt = await this.refreshToken(this.jwt, _Auth.tokenStrategy.refresh.whenScheduled);
          super.emit("updated", this.jwt);
          await this.save();
        } catch (error) {
          this.logger.error({ error }, "Error when refreshing jwt");
        }
      } else {
        this.logger.debug("Check token, still valid");
      }
    }, _Auth.tokenStrategy.refresh.interval);
  }
};
var Auth = _Auth;
Auth.authPageUrl = "https://app.tabbyml.com/account/device-token";
Auth.tokenStrategy = {
  polling: {
    // polling token after auth url generated
    interval: 5e3,
    // polling token every 5 seconds
    timeout: 5 * 60 * 1e3
    // stop polling after trying for 5 min
  },
  refresh: {
    // check token every 15 min, refresh token if it expires in 30 min
    interval: 15 * 60 * 1e3,
    beforeExpire: 30 * 60 * 1e3,
    whenLoaded: {
      // after token loaded from data store, refresh token if it is about to expire or has expired
      maxTry: 5,
      // keep loading time not too long
      retryDelay: 1e3
      // retry after 1 seconds
    },
    whenScheduled: {
      // if running until token is about to expire, refresh token as scheduled
      maxTry: 60,
      retryDelay: 30 * 1e3
      // retry after 30 seconds
    }
  }
};

// src/AgentConfig.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var defaultAgentConfig = {
  server: {
    endpoint: "http://localhost:8080"
  },
  logs: {
    level: "silent"
  },
  anonymousUsageTracking: {
    disable: false
  }
};

// src/CompletionCache.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// node_modules/lru-cache/dist/mjs/index.js
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var perf = typeof performance === "object" && performance && typeof performance.now === "function" ? performance : Date;
var warned = /* @__PURE__ */ new Set();
var PROCESS = typeof process_exports === "object" && !!process_exports ? process_exports : {};
var emitWarning2 = (msg, type2, code, fn) => {
  typeof PROCESS.emitWarning === "function" ? PROCESS.emitWarning(msg, type2, code, fn) : console.error(`[${code}] ${type2}: ${msg}`);
};
var AC = globalThis.AbortController;
var AS = globalThis.AbortSignal;
if (typeof AC === "undefined") {
  AS = class AbortSignal {
    constructor() {
      __publicField(this, "onabort");
      __publicField(this, "_onabort", []);
      __publicField(this, "reason");
      __publicField(this, "aborted", false);
    }
    addEventListener(_4, fn) {
      this._onabort.push(fn);
    }
  };
  AC = class AbortController {
    constructor() {
      __publicField(this, "signal", new AS());
      warnACPolyfill();
    }
    abort(reason) {
      if (this.signal.aborted)
        return;
      this.signal.reason = reason;
      this.signal.aborted = true;
      for (const fn of this.signal._onabort) {
        fn(reason);
      }
      this.signal.onabort?.(reason);
    }
  };
  let printACPolyfillWarning = PROCESS.env?.LRU_CACHE_IGNORE_AC_WARNING !== "1";
  const warnACPolyfill = () => {
    if (!printACPolyfillWarning)
      return;
    printACPolyfillWarning = false;
    emitWarning2("AbortController is not defined. If using lru-cache in node 14, load an AbortController polyfill from the `node-abort-controller` package. A minimal polyfill is provided for use by LRUCache.fetch(), but it should not be relied upon in other contexts (eg, passing it to other APIs that use AbortController/AbortSignal might have undesirable effects). You may disable this with LRU_CACHE_IGNORE_AC_WARNING=1 in the env.", "NO_ABORT_CONTROLLER", "ENOTSUP", warnACPolyfill);
  };
}
var shouldWarn = (code) => !warned.has(code);
var isPosInt = (n9) => n9 && n9 === Math.floor(n9) && n9 > 0 && isFinite(n9);
var getUintArray = (max) => !isPosInt(max) ? null : max <= Math.pow(2, 8) ? Uint8Array : max <= Math.pow(2, 16) ? Uint16Array : max <= Math.pow(2, 32) ? Uint32Array : max <= Number.MAX_SAFE_INTEGER ? ZeroArray : null;
var ZeroArray = class extends Array {
  constructor(size) {
    super(size);
    this.fill(0);
  }
};
var _constructing;
var _Stack = class {
  constructor(max, HeapCls) {
    __publicField(this, "heap");
    __publicField(this, "length");
    if (!__privateGet(_Stack, _constructing)) {
      throw new TypeError("instantiate Stack using Stack.create(n)");
    }
    this.heap = new HeapCls(max);
    this.length = 0;
  }
  static create(max) {
    const HeapCls = getUintArray(max);
    if (!HeapCls)
      return [];
    __privateSet(_Stack, _constructing, true);
    const s6 = new _Stack(max, HeapCls);
    __privateSet(_Stack, _constructing, false);
    return s6;
  }
  push(n9) {
    this.heap[this.length++] = n9;
  }
  pop() {
    return this.heap[--this.length];
  }
};
var Stack = _Stack;
_constructing = new WeakMap();
// private constructor
__privateAdd(Stack, _constructing, false);
var _max, _maxSize, _dispose, _disposeAfter, _fetchMethod, _size, _calculatedSize, _keyMap, _keyList, _valList, _next, _prev, _head, _tail, _free, _disposed, _sizes, _starts, _ttls, _hasDispose, _hasFetchMethod, _hasDisposeAfter, _initializeTTLTracking, initializeTTLTracking_fn, _updateItemAge, _statusTTL, _setItemTTL, _isStale, _initializeSizeTracking, initializeSizeTracking_fn, _removeItemSize, _addItemSize, _requireSize, _indexes, indexes_fn, _rindexes, rindexes_fn, _isValidIndex, isValidIndex_fn, _evict, evict_fn, _backgroundFetch, backgroundFetch_fn, _isBackgroundFetch, isBackgroundFetch_fn, _connect, connect_fn, _moveToTail, moveToTail_fn;
var _LRUCache = class {
  constructor(options) {
    __privateAdd(this, _initializeTTLTracking);
    __privateAdd(this, _initializeSizeTracking);
    __privateAdd(this, _indexes);
    __privateAdd(this, _rindexes);
    __privateAdd(this, _isValidIndex);
    __privateAdd(this, _evict);
    __privateAdd(this, _backgroundFetch);
    __privateAdd(this, _isBackgroundFetch);
    __privateAdd(this, _connect);
    __privateAdd(this, _moveToTail);
    // properties coming in from the options of these, only max and maxSize
    // really *need* to be protected. The rest can be modified, as they just
    // set defaults for various methods.
    __privateAdd(this, _max, void 0);
    __privateAdd(this, _maxSize, void 0);
    __privateAdd(this, _dispose, void 0);
    __privateAdd(this, _disposeAfter, void 0);
    __privateAdd(this, _fetchMethod, void 0);
    /**
     * {@link LRUCache.OptionsBase.ttl}
     */
    __publicField(this, "ttl");
    /**
     * {@link LRUCache.OptionsBase.ttlResolution}
     */
    __publicField(this, "ttlResolution");
    /**
     * {@link LRUCache.OptionsBase.ttlAutopurge}
     */
    __publicField(this, "ttlAutopurge");
    /**
     * {@link LRUCache.OptionsBase.updateAgeOnGet}
     */
    __publicField(this, "updateAgeOnGet");
    /**
     * {@link LRUCache.OptionsBase.updateAgeOnHas}
     */
    __publicField(this, "updateAgeOnHas");
    /**
     * {@link LRUCache.OptionsBase.allowStale}
     */
    __publicField(this, "allowStale");
    /**
     * {@link LRUCache.OptionsBase.noDisposeOnSet}
     */
    __publicField(this, "noDisposeOnSet");
    /**
     * {@link LRUCache.OptionsBase.noUpdateTTL}
     */
    __publicField(this, "noUpdateTTL");
    /**
     * {@link LRUCache.OptionsBase.maxEntrySize}
     */
    __publicField(this, "maxEntrySize");
    /**
     * {@link LRUCache.OptionsBase.sizeCalculation}
     */
    __publicField(this, "sizeCalculation");
    /**
     * {@link LRUCache.OptionsBase.noDeleteOnFetchRejection}
     */
    __publicField(this, "noDeleteOnFetchRejection");
    /**
     * {@link LRUCache.OptionsBase.noDeleteOnStaleGet}
     */
    __publicField(this, "noDeleteOnStaleGet");
    /**
     * {@link LRUCache.OptionsBase.allowStaleOnFetchAbort}
     */
    __publicField(this, "allowStaleOnFetchAbort");
    /**
     * {@link LRUCache.OptionsBase.allowStaleOnFetchRejection}
     */
    __publicField(this, "allowStaleOnFetchRejection");
    /**
     * {@link LRUCache.OptionsBase.ignoreFetchAbort}
     */
    __publicField(this, "ignoreFetchAbort");
    // computed properties
    __privateAdd(this, _size, void 0);
    __privateAdd(this, _calculatedSize, void 0);
    __privateAdd(this, _keyMap, void 0);
    __privateAdd(this, _keyList, void 0);
    __privateAdd(this, _valList, void 0);
    __privateAdd(this, _next, void 0);
    __privateAdd(this, _prev, void 0);
    __privateAdd(this, _head, void 0);
    __privateAdd(this, _tail, void 0);
    __privateAdd(this, _free, void 0);
    __privateAdd(this, _disposed, void 0);
    __privateAdd(this, _sizes, void 0);
    __privateAdd(this, _starts, void 0);
    __privateAdd(this, _ttls, void 0);
    __privateAdd(this, _hasDispose, void 0);
    __privateAdd(this, _hasFetchMethod, void 0);
    __privateAdd(this, _hasDisposeAfter, void 0);
    // conditionally set private methods related to TTL
    __privateAdd(this, _updateItemAge, () => {
    });
    __privateAdd(this, _statusTTL, () => {
    });
    __privateAdd(this, _setItemTTL, () => {
    });
    /* c8 ignore stop */
    __privateAdd(this, _isStale, () => false);
    __privateAdd(this, _removeItemSize, (_i) => {
    });
    __privateAdd(this, _addItemSize, (_i, _s, _st) => {
    });
    __privateAdd(this, _requireSize, (_k, _v, size, sizeCalculation) => {
      if (size || sizeCalculation) {
        throw new TypeError("cannot set size without setting maxSize or maxEntrySize on cache");
      }
      return 0;
    });
    const { max = 0, ttl, ttlResolution = 1, ttlAutopurge, updateAgeOnGet, updateAgeOnHas, allowStale, dispose, disposeAfter, noDisposeOnSet, noUpdateTTL, maxSize = 0, maxEntrySize = 0, sizeCalculation, fetchMethod, noDeleteOnFetchRejection, noDeleteOnStaleGet, allowStaleOnFetchRejection, allowStaleOnFetchAbort, ignoreFetchAbort } = options;
    if (max !== 0 && !isPosInt(max)) {
      throw new TypeError("max option must be a nonnegative integer");
    }
    const UintArray = max ? getUintArray(max) : Array;
    if (!UintArray) {
      throw new Error("invalid max value: " + max);
    }
    __privateSet(this, _max, max);
    __privateSet(this, _maxSize, maxSize);
    this.maxEntrySize = maxEntrySize || __privateGet(this, _maxSize);
    this.sizeCalculation = sizeCalculation;
    if (this.sizeCalculation) {
      if (!__privateGet(this, _maxSize) && !this.maxEntrySize) {
        throw new TypeError("cannot set sizeCalculation without setting maxSize or maxEntrySize");
      }
      if (typeof this.sizeCalculation !== "function") {
        throw new TypeError("sizeCalculation set to non-function");
      }
    }
    if (fetchMethod !== void 0 && typeof fetchMethod !== "function") {
      throw new TypeError("fetchMethod must be a function if specified");
    }
    __privateSet(this, _fetchMethod, fetchMethod);
    __privateSet(this, _hasFetchMethod, !!fetchMethod);
    __privateSet(this, _keyMap, /* @__PURE__ */ new Map());
    __privateSet(this, _keyList, new Array(max).fill(void 0));
    __privateSet(this, _valList, new Array(max).fill(void 0));
    __privateSet(this, _next, new UintArray(max));
    __privateSet(this, _prev, new UintArray(max));
    __privateSet(this, _head, 0);
    __privateSet(this, _tail, 0);
    __privateSet(this, _free, Stack.create(max));
    __privateSet(this, _size, 0);
    __privateSet(this, _calculatedSize, 0);
    if (typeof dispose === "function") {
      __privateSet(this, _dispose, dispose);
    }
    if (typeof disposeAfter === "function") {
      __privateSet(this, _disposeAfter, disposeAfter);
      __privateSet(this, _disposed, []);
    } else {
      __privateSet(this, _disposeAfter, void 0);
      __privateSet(this, _disposed, void 0);
    }
    __privateSet(this, _hasDispose, !!__privateGet(this, _dispose));
    __privateSet(this, _hasDisposeAfter, !!__privateGet(this, _disposeAfter));
    this.noDisposeOnSet = !!noDisposeOnSet;
    this.noUpdateTTL = !!noUpdateTTL;
    this.noDeleteOnFetchRejection = !!noDeleteOnFetchRejection;
    this.allowStaleOnFetchRejection = !!allowStaleOnFetchRejection;
    this.allowStaleOnFetchAbort = !!allowStaleOnFetchAbort;
    this.ignoreFetchAbort = !!ignoreFetchAbort;
    if (this.maxEntrySize !== 0) {
      if (__privateGet(this, _maxSize) !== 0) {
        if (!isPosInt(__privateGet(this, _maxSize))) {
          throw new TypeError("maxSize must be a positive integer if specified");
        }
      }
      if (!isPosInt(this.maxEntrySize)) {
        throw new TypeError("maxEntrySize must be a positive integer if specified");
      }
      __privateMethod(this, _initializeSizeTracking, initializeSizeTracking_fn).call(this);
    }
    this.allowStale = !!allowStale;
    this.noDeleteOnStaleGet = !!noDeleteOnStaleGet;
    this.updateAgeOnGet = !!updateAgeOnGet;
    this.updateAgeOnHas = !!updateAgeOnHas;
    this.ttlResolution = isPosInt(ttlResolution) || ttlResolution === 0 ? ttlResolution : 1;
    this.ttlAutopurge = !!ttlAutopurge;
    this.ttl = ttl || 0;
    if (this.ttl) {
      if (!isPosInt(this.ttl)) {
        throw new TypeError("ttl must be a positive integer if specified");
      }
      __privateMethod(this, _initializeTTLTracking, initializeTTLTracking_fn).call(this);
    }
    if (__privateGet(this, _max) === 0 && this.ttl === 0 && __privateGet(this, _maxSize) === 0) {
      throw new TypeError("At least one of max, maxSize, or ttl is required");
    }
    if (!this.ttlAutopurge && !__privateGet(this, _max) && !__privateGet(this, _maxSize)) {
      const code = "LRU_CACHE_UNBOUNDED";
      if (shouldWarn(code)) {
        warned.add(code);
        const msg = "TTL caching without ttlAutopurge, max, or maxSize can result in unbounded memory consumption.";
        emitWarning2(msg, "UnboundedCacheWarning", code, _LRUCache);
      }
    }
  }
  /**
   * Do not call this method unless you need to inspect the
   * inner workings of the cache.  If anything returned by this
   * object is modified in any way, strange breakage may occur.
   *
   * These fields are private for a reason!
   *
   * @internal
   */
  static unsafeExposeInternals(c7) {
    return {
      // properties
      starts: __privateGet(c7, _starts),
      ttls: __privateGet(c7, _ttls),
      sizes: __privateGet(c7, _sizes),
      keyMap: __privateGet(c7, _keyMap),
      keyList: __privateGet(c7, _keyList),
      valList: __privateGet(c7, _valList),
      next: __privateGet(c7, _next),
      prev: __privateGet(c7, _prev),
      get head() {
        return __privateGet(c7, _head);
      },
      get tail() {
        return __privateGet(c7, _tail);
      },
      free: __privateGet(c7, _free),
      // methods
      isBackgroundFetch: (p7) => {
        var _a;
        return __privateMethod(_a = c7, _isBackgroundFetch, isBackgroundFetch_fn).call(_a, p7);
      },
      backgroundFetch: (k4, index, options, context) => {
        var _a;
        return __privateMethod(_a = c7, _backgroundFetch, backgroundFetch_fn).call(_a, k4, index, options, context);
      },
      moveToTail: (index) => {
        var _a;
        return __privateMethod(_a = c7, _moveToTail, moveToTail_fn).call(_a, index);
      },
      indexes: (options) => {
        var _a;
        return __privateMethod(_a = c7, _indexes, indexes_fn).call(_a, options);
      },
      rindexes: (options) => {
        var _a;
        return __privateMethod(_a = c7, _rindexes, rindexes_fn).call(_a, options);
      },
      isStale: (index) => {
        var _a;
        return __privateGet(_a = c7, _isStale).call(_a, index);
      }
    };
  }
  // Protected read-only members
  /**
   * {@link LRUCache.OptionsBase.max} (read-only)
   */
  get max() {
    return __privateGet(this, _max);
  }
  /**
   * {@link LRUCache.OptionsBase.maxSize} (read-only)
   */
  get maxSize() {
    return __privateGet(this, _maxSize);
  }
  /**
   * The total computed size of items in the cache (read-only)
   */
  get calculatedSize() {
    return __privateGet(this, _calculatedSize);
  }
  /**
   * The number of items stored in the cache (read-only)
   */
  get size() {
    return __privateGet(this, _size);
  }
  /**
   * {@link LRUCache.OptionsBase.fetchMethod} (read-only)
   */
  get fetchMethod() {
    return __privateGet(this, _fetchMethod);
  }
  /**
   * {@link LRUCache.OptionsBase.dispose} (read-only)
   */
  get dispose() {
    return __privateGet(this, _dispose);
  }
  /**
   * {@link LRUCache.OptionsBase.disposeAfter} (read-only)
   */
  get disposeAfter() {
    return __privateGet(this, _disposeAfter);
  }
  /**
   * Return the remaining TTL time for a given entry key
   */
  getRemainingTTL(key) {
    return __privateGet(this, _keyMap).has(key) ? Infinity : 0;
  }
  /**
   * Return a generator yielding `[key, value]` pairs,
   * in order from most recently used to least recently used.
   */
  *entries() {
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
      if (__privateGet(this, _valList)[i7] !== void 0 && __privateGet(this, _keyList)[i7] !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield [__privateGet(this, _keyList)[i7], __privateGet(this, _valList)[i7]];
      }
    }
  }
  /**
   * Inverse order version of {@link LRUCache.entries}
   *
   * Return a generator yielding `[key, value]` pairs,
   * in order from least recently used to most recently used.
   */
  *rentries() {
    for (const i7 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
      if (__privateGet(this, _valList)[i7] !== void 0 && __privateGet(this, _keyList)[i7] !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield [__privateGet(this, _keyList)[i7], __privateGet(this, _valList)[i7]];
      }
    }
  }
  /**
   * Return a generator yielding the keys in the cache,
   * in order from most recently used to least recently used.
   */
  *keys() {
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
      const k4 = __privateGet(this, _keyList)[i7];
      if (k4 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield k4;
      }
    }
  }
  /**
   * Inverse order version of {@link LRUCache.keys}
   *
   * Return a generator yielding the keys in the cache,
   * in order from least recently used to most recently used.
   */
  *rkeys() {
    for (const i7 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
      const k4 = __privateGet(this, _keyList)[i7];
      if (k4 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield k4;
      }
    }
  }
  /**
   * Return a generator yielding the values in the cache,
   * in order from most recently used to least recently used.
   */
  *values() {
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
      const v7 = __privateGet(this, _valList)[i7];
      if (v7 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield __privateGet(this, _valList)[i7];
      }
    }
  }
  /**
   * Inverse order version of {@link LRUCache.values}
   *
   * Return a generator yielding the values in the cache,
   * in order from least recently used to most recently used.
   */
  *rvalues() {
    for (const i7 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
      const v7 = __privateGet(this, _valList)[i7];
      if (v7 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i7])) {
        yield __privateGet(this, _valList)[i7];
      }
    }
  }
  /**
   * Iterating over the cache itself yields the same results as
   * {@link LRUCache.entries}
   */
  [Symbol.iterator]() {
    return this.entries();
  }
  /**
   * Find a value for which the supplied fn method returns a truthy value,
   * similar to Array.find().  fn is called as fn(value, key, cache).
   */
  find(fn, getOptions = {}) {
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
      const v7 = __privateGet(this, _valList)[i7];
      const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) ? v7.__staleWhileFetching : v7;
      if (value === void 0)
        continue;
      if (fn(value, __privateGet(this, _keyList)[i7], this)) {
        return this.get(__privateGet(this, _keyList)[i7], getOptions);
      }
    }
  }
  /**
   * Call the supplied function on each item in the cache, in order from
   * most recently used to least recently used.  fn is called as
   * fn(value, key, cache).  Does not update age or recenty of use.
   * Does not iterate over stale values.
   */
  forEach(fn, thisp = this) {
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
      const v7 = __privateGet(this, _valList)[i7];
      const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) ? v7.__staleWhileFetching : v7;
      if (value === void 0)
        continue;
      fn.call(thisp, value, __privateGet(this, _keyList)[i7], this);
    }
  }
  /**
   * The same as {@link LRUCache.forEach} but items are iterated over in
   * reverse order.  (ie, less recently used items are iterated over first.)
   */
  rforEach(fn, thisp = this) {
    for (const i7 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
      const v7 = __privateGet(this, _valList)[i7];
      const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) ? v7.__staleWhileFetching : v7;
      if (value === void 0)
        continue;
      fn.call(thisp, value, __privateGet(this, _keyList)[i7], this);
    }
  }
  /**
   * Delete any stale entries. Returns true if anything was removed,
   * false otherwise.
   */
  purgeStale() {
    let deleted = false;
    for (const i7 of __privateMethod(this, _rindexes, rindexes_fn).call(this, { allowStale: true })) {
      if (__privateGet(this, _isStale).call(this, i7)) {
        this.delete(__privateGet(this, _keyList)[i7]);
        deleted = true;
      }
    }
    return deleted;
  }
  /**
   * Return an array of [key, {@link LRUCache.Entry}] tuples which can be
   * passed to cache.load()
   */
  dump() {
    const arr = [];
    for (const i7 of __privateMethod(this, _indexes, indexes_fn).call(this, { allowStale: true })) {
      const key = __privateGet(this, _keyList)[i7];
      const v7 = __privateGet(this, _valList)[i7];
      const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) ? v7.__staleWhileFetching : v7;
      if (value === void 0 || key === void 0)
        continue;
      const entry = { value };
      if (__privateGet(this, _ttls) && __privateGet(this, _starts)) {
        entry.ttl = __privateGet(this, _ttls)[i7];
        const age = perf.now() - __privateGet(this, _starts)[i7];
        entry.start = Math.floor(Date.now() - age);
      }
      if (__privateGet(this, _sizes)) {
        entry.size = __privateGet(this, _sizes)[i7];
      }
      arr.unshift([key, entry]);
    }
    return arr;
  }
  /**
   * Reset the cache and load in the items in entries in the order listed.
   * Note that the shape of the resulting cache may be different if the
   * same options are not used in both caches.
   */
  load(arr) {
    this.clear();
    for (const [key, entry] of arr) {
      if (entry.start) {
        const age = Date.now() - entry.start;
        entry.start = perf.now() - age;
      }
      this.set(key, entry.value, entry);
    }
  }
  /**
   * Add a value to the cache.
   *
   * Note: if `undefined` is specified as a value, this is an alias for
   * {@link LRUCache#delete}
   */
  set(k4, v7, setOptions = {}) {
    var _a, _b;
    if (v7 === void 0) {
      this.delete(k4);
      return this;
    }
    const { ttl = this.ttl, start, noDisposeOnSet = this.noDisposeOnSet, sizeCalculation = this.sizeCalculation, status } = setOptions;
    let { noUpdateTTL = this.noUpdateTTL } = setOptions;
    const size = __privateGet(this, _requireSize).call(this, k4, v7, setOptions.size || 0, sizeCalculation);
    if (this.maxEntrySize && size > this.maxEntrySize) {
      if (status) {
        status.set = "miss";
        status.maxEntrySizeExceeded = true;
      }
      this.delete(k4);
      return this;
    }
    let index = __privateGet(this, _size) === 0 ? void 0 : __privateGet(this, _keyMap).get(k4);
    if (index === void 0) {
      index = __privateGet(this, _size) === 0 ? __privateGet(this, _tail) : __privateGet(this, _free).length !== 0 ? __privateGet(this, _free).pop() : __privateGet(this, _size) === __privateGet(this, _max) ? __privateMethod(this, _evict, evict_fn).call(this, false) : __privateGet(this, _size);
      __privateGet(this, _keyList)[index] = k4;
      __privateGet(this, _valList)[index] = v7;
      __privateGet(this, _keyMap).set(k4, index);
      __privateGet(this, _next)[__privateGet(this, _tail)] = index;
      __privateGet(this, _prev)[index] = __privateGet(this, _tail);
      __privateSet(this, _tail, index);
      __privateWrapper(this, _size)._++;
      __privateGet(this, _addItemSize).call(this, index, size, status);
      if (status)
        status.set = "add";
      noUpdateTTL = false;
    } else {
      __privateMethod(this, _moveToTail, moveToTail_fn).call(this, index);
      const oldVal = __privateGet(this, _valList)[index];
      if (v7 !== oldVal) {
        if (__privateGet(this, _hasFetchMethod) && __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, oldVal)) {
          oldVal.__abortController.abort(new Error("replaced"));
        } else if (!noDisposeOnSet) {
          if (__privateGet(this, _hasDispose)) {
            (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, oldVal, k4, "set");
          }
          if (__privateGet(this, _hasDisposeAfter)) {
            __privateGet(this, _disposed)?.push([oldVal, k4, "set"]);
          }
        }
        __privateGet(this, _removeItemSize).call(this, index);
        __privateGet(this, _addItemSize).call(this, index, size, status);
        __privateGet(this, _valList)[index] = v7;
        if (status) {
          status.set = "replace";
          const oldValue = oldVal && __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, oldVal) ? oldVal.__staleWhileFetching : oldVal;
          if (oldValue !== void 0)
            status.oldValue = oldValue;
        }
      } else if (status) {
        status.set = "update";
      }
    }
    if (ttl !== 0 && !__privateGet(this, _ttls)) {
      __privateMethod(this, _initializeTTLTracking, initializeTTLTracking_fn).call(this);
    }
    if (__privateGet(this, _ttls)) {
      if (!noUpdateTTL) {
        __privateGet(this, _setItemTTL).call(this, index, ttl, start);
      }
      if (status)
        __privateGet(this, _statusTTL).call(this, status, index);
    }
    if (!noDisposeOnSet && __privateGet(this, _hasDisposeAfter) && __privateGet(this, _disposed)) {
      const dt = __privateGet(this, _disposed);
      let task;
      while (task = dt?.shift()) {
        (_b = __privateGet(this, _disposeAfter)) == null ? void 0 : _b.call(this, ...task);
      }
    }
    return this;
  }
  /**
   * Evict the least recently used item, returning its value or
   * `undefined` if cache is empty.
   */
  pop() {
    var _a;
    try {
      while (__privateGet(this, _size)) {
        const val = __privateGet(this, _valList)[__privateGet(this, _head)];
        __privateMethod(this, _evict, evict_fn).call(this, true);
        if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, val)) {
          if (val.__staleWhileFetching) {
            return val.__staleWhileFetching;
          }
        } else if (val !== void 0) {
          return val;
        }
      }
    } finally {
      if (__privateGet(this, _hasDisposeAfter) && __privateGet(this, _disposed)) {
        const dt = __privateGet(this, _disposed);
        let task;
        while (task = dt?.shift()) {
          (_a = __privateGet(this, _disposeAfter)) == null ? void 0 : _a.call(this, ...task);
        }
      }
    }
  }
  /**
   * Check if a key is in the cache, without updating the recency of use.
   * Will return false if the item is stale, even though it is technically
   * in the cache.
   *
   * Will not update item age unless
   * {@link LRUCache.OptionsBase.updateAgeOnHas} is set.
   */
  has(k4, hasOptions = {}) {
    const { updateAgeOnHas = this.updateAgeOnHas, status } = hasOptions;
    const index = __privateGet(this, _keyMap).get(k4);
    if (index !== void 0) {
      const v7 = __privateGet(this, _valList)[index];
      if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) && v7.__staleWhileFetching === void 0) {
        return false;
      }
      if (!__privateGet(this, _isStale).call(this, index)) {
        if (updateAgeOnHas) {
          __privateGet(this, _updateItemAge).call(this, index);
        }
        if (status) {
          status.has = "hit";
          __privateGet(this, _statusTTL).call(this, status, index);
        }
        return true;
      } else if (status) {
        status.has = "stale";
        __privateGet(this, _statusTTL).call(this, status, index);
      }
    } else if (status) {
      status.has = "miss";
    }
    return false;
  }
  /**
   * Like {@link LRUCache#get} but doesn't update recency or delete stale
   * items.
   *
   * Returns `undefined` if the item is stale, unless
   * {@link LRUCache.OptionsBase.allowStale} is set.
   */
  peek(k4, peekOptions = {}) {
    const { allowStale = this.allowStale } = peekOptions;
    const index = __privateGet(this, _keyMap).get(k4);
    if (index !== void 0 && (allowStale || !__privateGet(this, _isStale).call(this, index))) {
      const v7 = __privateGet(this, _valList)[index];
      return __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7) ? v7.__staleWhileFetching : v7;
    }
  }
  async fetch(k4, fetchOptions = {}) {
    const {
      // get options
      allowStale = this.allowStale,
      updateAgeOnGet = this.updateAgeOnGet,
      noDeleteOnStaleGet = this.noDeleteOnStaleGet,
      // set options
      ttl = this.ttl,
      noDisposeOnSet = this.noDisposeOnSet,
      size = 0,
      sizeCalculation = this.sizeCalculation,
      noUpdateTTL = this.noUpdateTTL,
      // fetch exclusive options
      noDeleteOnFetchRejection = this.noDeleteOnFetchRejection,
      allowStaleOnFetchRejection = this.allowStaleOnFetchRejection,
      ignoreFetchAbort = this.ignoreFetchAbort,
      allowStaleOnFetchAbort = this.allowStaleOnFetchAbort,
      context,
      forceRefresh = false,
      status,
      signal
    } = fetchOptions;
    if (!__privateGet(this, _hasFetchMethod)) {
      if (status)
        status.fetch = "get";
      return this.get(k4, {
        allowStale,
        updateAgeOnGet,
        noDeleteOnStaleGet,
        status
      });
    }
    const options = {
      allowStale,
      updateAgeOnGet,
      noDeleteOnStaleGet,
      ttl,
      noDisposeOnSet,
      size,
      sizeCalculation,
      noUpdateTTL,
      noDeleteOnFetchRejection,
      allowStaleOnFetchRejection,
      allowStaleOnFetchAbort,
      ignoreFetchAbort,
      status,
      signal
    };
    let index = __privateGet(this, _keyMap).get(k4);
    if (index === void 0) {
      if (status)
        status.fetch = "miss";
      const p7 = __privateMethod(this, _backgroundFetch, backgroundFetch_fn).call(this, k4, index, options, context);
      return p7.__returned = p7;
    } else {
      const v7 = __privateGet(this, _valList)[index];
      if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
        const stale = allowStale && v7.__staleWhileFetching !== void 0;
        if (status) {
          status.fetch = "inflight";
          if (stale)
            status.returnedStale = true;
        }
        return stale ? v7.__staleWhileFetching : v7.__returned = v7;
      }
      const isStale = __privateGet(this, _isStale).call(this, index);
      if (!forceRefresh && !isStale) {
        if (status)
          status.fetch = "hit";
        __privateMethod(this, _moveToTail, moveToTail_fn).call(this, index);
        if (updateAgeOnGet) {
          __privateGet(this, _updateItemAge).call(this, index);
        }
        if (status)
          __privateGet(this, _statusTTL).call(this, status, index);
        return v7;
      }
      const p7 = __privateMethod(this, _backgroundFetch, backgroundFetch_fn).call(this, k4, index, options, context);
      const hasStale = p7.__staleWhileFetching !== void 0;
      const staleVal = hasStale && allowStale;
      if (status) {
        status.fetch = isStale ? "stale" : "refresh";
        if (staleVal && isStale)
          status.returnedStale = true;
      }
      return staleVal ? p7.__staleWhileFetching : p7.__returned = p7;
    }
  }
  /**
   * Return a value from the cache. Will update the recency of the cache
   * entry found.
   *
   * If the key is not found, get() will return `undefined`.
   */
  get(k4, getOptions = {}) {
    const { allowStale = this.allowStale, updateAgeOnGet = this.updateAgeOnGet, noDeleteOnStaleGet = this.noDeleteOnStaleGet, status } = getOptions;
    const index = __privateGet(this, _keyMap).get(k4);
    if (index !== void 0) {
      const value = __privateGet(this, _valList)[index];
      const fetching = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, value);
      if (status)
        __privateGet(this, _statusTTL).call(this, status, index);
      if (__privateGet(this, _isStale).call(this, index)) {
        if (status)
          status.get = "stale";
        if (!fetching) {
          if (!noDeleteOnStaleGet) {
            this.delete(k4);
          }
          if (status && allowStale)
            status.returnedStale = true;
          return allowStale ? value : void 0;
        } else {
          if (status && allowStale && value.__staleWhileFetching !== void 0) {
            status.returnedStale = true;
          }
          return allowStale ? value.__staleWhileFetching : void 0;
        }
      } else {
        if (status)
          status.get = "hit";
        if (fetching) {
          return value.__staleWhileFetching;
        }
        __privateMethod(this, _moveToTail, moveToTail_fn).call(this, index);
        if (updateAgeOnGet) {
          __privateGet(this, _updateItemAge).call(this, index);
        }
        return value;
      }
    } else if (status) {
      status.get = "miss";
    }
  }
  /**
   * Deletes a key out of the cache.
   * Returns true if the key was deleted, false otherwise.
   */
  delete(k4) {
    var _a, _b;
    let deleted = false;
    if (__privateGet(this, _size) !== 0) {
      const index = __privateGet(this, _keyMap).get(k4);
      if (index !== void 0) {
        deleted = true;
        if (__privateGet(this, _size) === 1) {
          this.clear();
        } else {
          __privateGet(this, _removeItemSize).call(this, index);
          const v7 = __privateGet(this, _valList)[index];
          if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
            v7.__abortController.abort(new Error("deleted"));
          } else if (__privateGet(this, _hasDispose) || __privateGet(this, _hasDisposeAfter)) {
            if (__privateGet(this, _hasDispose)) {
              (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v7, k4, "delete");
            }
            if (__privateGet(this, _hasDisposeAfter)) {
              __privateGet(this, _disposed)?.push([v7, k4, "delete"]);
            }
          }
          __privateGet(this, _keyMap).delete(k4);
          __privateGet(this, _keyList)[index] = void 0;
          __privateGet(this, _valList)[index] = void 0;
          if (index === __privateGet(this, _tail)) {
            __privateSet(this, _tail, __privateGet(this, _prev)[index]);
          } else if (index === __privateGet(this, _head)) {
            __privateSet(this, _head, __privateGet(this, _next)[index]);
          } else {
            __privateGet(this, _next)[__privateGet(this, _prev)[index]] = __privateGet(this, _next)[index];
            __privateGet(this, _prev)[__privateGet(this, _next)[index]] = __privateGet(this, _prev)[index];
          }
          __privateWrapper(this, _size)._--;
          __privateGet(this, _free).push(index);
        }
      }
    }
    if (__privateGet(this, _hasDisposeAfter) && __privateGet(this, _disposed)?.length) {
      const dt = __privateGet(this, _disposed);
      let task;
      while (task = dt?.shift()) {
        (_b = __privateGet(this, _disposeAfter)) == null ? void 0 : _b.call(this, ...task);
      }
    }
    return deleted;
  }
  /**
   * Clear the cache entirely, throwing away all values.
   */
  clear() {
    var _a, _b;
    for (const index of __privateMethod(this, _rindexes, rindexes_fn).call(this, { allowStale: true })) {
      const v7 = __privateGet(this, _valList)[index];
      if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
        v7.__abortController.abort(new Error("deleted"));
      } else {
        const k4 = __privateGet(this, _keyList)[index];
        if (__privateGet(this, _hasDispose)) {
          (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v7, k4, "delete");
        }
        if (__privateGet(this, _hasDisposeAfter)) {
          __privateGet(this, _disposed)?.push([v7, k4, "delete"]);
        }
      }
    }
    __privateGet(this, _keyMap).clear();
    __privateGet(this, _valList).fill(void 0);
    __privateGet(this, _keyList).fill(void 0);
    if (__privateGet(this, _ttls) && __privateGet(this, _starts)) {
      __privateGet(this, _ttls).fill(0);
      __privateGet(this, _starts).fill(0);
    }
    if (__privateGet(this, _sizes)) {
      __privateGet(this, _sizes).fill(0);
    }
    __privateSet(this, _head, 0);
    __privateSet(this, _tail, 0);
    __privateGet(this, _free).length = 0;
    __privateSet(this, _calculatedSize, 0);
    __privateSet(this, _size, 0);
    if (__privateGet(this, _hasDisposeAfter) && __privateGet(this, _disposed)) {
      const dt = __privateGet(this, _disposed);
      let task;
      while (task = dt?.shift()) {
        (_b = __privateGet(this, _disposeAfter)) == null ? void 0 : _b.call(this, ...task);
      }
    }
  }
};
var LRUCache = _LRUCache;
_max = new WeakMap();
_maxSize = new WeakMap();
_dispose = new WeakMap();
_disposeAfter = new WeakMap();
_fetchMethod = new WeakMap();
_size = new WeakMap();
_calculatedSize = new WeakMap();
_keyMap = new WeakMap();
_keyList = new WeakMap();
_valList = new WeakMap();
_next = new WeakMap();
_prev = new WeakMap();
_head = new WeakMap();
_tail = new WeakMap();
_free = new WeakMap();
_disposed = new WeakMap();
_sizes = new WeakMap();
_starts = new WeakMap();
_ttls = new WeakMap();
_hasDispose = new WeakMap();
_hasFetchMethod = new WeakMap();
_hasDisposeAfter = new WeakMap();
_initializeTTLTracking = new WeakSet();
initializeTTLTracking_fn = function() {
  const ttls = new ZeroArray(__privateGet(this, _max));
  const starts = new ZeroArray(__privateGet(this, _max));
  __privateSet(this, _ttls, ttls);
  __privateSet(this, _starts, starts);
  __privateSet(this, _setItemTTL, (index, ttl, start = perf.now()) => {
    starts[index] = ttl !== 0 ? start : 0;
    ttls[index] = ttl;
    if (ttl !== 0 && this.ttlAutopurge) {
      const t9 = setTimeout(() => {
        if (__privateGet(this, _isStale).call(this, index)) {
          this.delete(__privateGet(this, _keyList)[index]);
        }
      }, ttl + 1);
      if (t9.unref) {
        t9.unref();
      }
    }
  });
  __privateSet(this, _updateItemAge, (index) => {
    starts[index] = ttls[index] !== 0 ? perf.now() : 0;
  });
  __privateSet(this, _statusTTL, (status, index) => {
    if (ttls[index]) {
      const ttl = ttls[index];
      const start = starts[index];
      status.ttl = ttl;
      status.start = start;
      status.now = cachedNow || getNow();
      const age = status.now - start;
      status.remainingTTL = ttl - age;
    }
  });
  let cachedNow = 0;
  const getNow = () => {
    const n9 = perf.now();
    if (this.ttlResolution > 0) {
      cachedNow = n9;
      const t9 = setTimeout(() => cachedNow = 0, this.ttlResolution);
      if (t9.unref) {
        t9.unref();
      }
    }
    return n9;
  };
  this.getRemainingTTL = (key) => {
    const index = __privateGet(this, _keyMap).get(key);
    if (index === void 0) {
      return 0;
    }
    const ttl = ttls[index];
    const start = starts[index];
    if (ttl === 0 || start === 0) {
      return Infinity;
    }
    const age = (cachedNow || getNow()) - start;
    return ttl - age;
  };
  __privateSet(this, _isStale, (index) => {
    return ttls[index] !== 0 && starts[index] !== 0 && (cachedNow || getNow()) - starts[index] > ttls[index];
  });
};
_updateItemAge = new WeakMap();
_statusTTL = new WeakMap();
_setItemTTL = new WeakMap();
_isStale = new WeakMap();
_initializeSizeTracking = new WeakSet();
initializeSizeTracking_fn = function() {
  const sizes = new ZeroArray(__privateGet(this, _max));
  __privateSet(this, _calculatedSize, 0);
  __privateSet(this, _sizes, sizes);
  __privateSet(this, _removeItemSize, (index) => {
    __privateSet(this, _calculatedSize, __privateGet(this, _calculatedSize) - sizes[index]);
    sizes[index] = 0;
  });
  __privateSet(this, _requireSize, (k4, v7, size, sizeCalculation) => {
    if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
      return 0;
    }
    if (!isPosInt(size)) {
      if (sizeCalculation) {
        if (typeof sizeCalculation !== "function") {
          throw new TypeError("sizeCalculation must be a function");
        }
        size = sizeCalculation(v7, k4);
        if (!isPosInt(size)) {
          throw new TypeError("sizeCalculation return invalid (expect positive integer)");
        }
      } else {
        throw new TypeError("invalid size value (must be positive integer). When maxSize or maxEntrySize is used, sizeCalculation or size must be set.");
      }
    }
    return size;
  });
  __privateSet(this, _addItemSize, (index, size, status) => {
    sizes[index] = size;
    if (__privateGet(this, _maxSize)) {
      const maxSize = __privateGet(this, _maxSize) - sizes[index];
      while (__privateGet(this, _calculatedSize) > maxSize) {
        __privateMethod(this, _evict, evict_fn).call(this, true);
      }
    }
    __privateSet(this, _calculatedSize, __privateGet(this, _calculatedSize) + sizes[index]);
    if (status) {
      status.entrySize = size;
      status.totalCalculatedSize = __privateGet(this, _calculatedSize);
    }
  });
};
_removeItemSize = new WeakMap();
_addItemSize = new WeakMap();
_requireSize = new WeakMap();
_indexes = new WeakSet();
indexes_fn = function* ({ allowStale = this.allowStale } = {}) {
  if (__privateGet(this, _size)) {
    for (let i7 = __privateGet(this, _tail); true; ) {
      if (!__privateMethod(this, _isValidIndex, isValidIndex_fn).call(this, i7)) {
        break;
      }
      if (allowStale || !__privateGet(this, _isStale).call(this, i7)) {
        yield i7;
      }
      if (i7 === __privateGet(this, _head)) {
        break;
      } else {
        i7 = __privateGet(this, _prev)[i7];
      }
    }
  }
};
_rindexes = new WeakSet();
rindexes_fn = function* ({ allowStale = this.allowStale } = {}) {
  if (__privateGet(this, _size)) {
    for (let i7 = __privateGet(this, _head); true; ) {
      if (!__privateMethod(this, _isValidIndex, isValidIndex_fn).call(this, i7)) {
        break;
      }
      if (allowStale || !__privateGet(this, _isStale).call(this, i7)) {
        yield i7;
      }
      if (i7 === __privateGet(this, _tail)) {
        break;
      } else {
        i7 = __privateGet(this, _next)[i7];
      }
    }
  }
};
_isValidIndex = new WeakSet();
isValidIndex_fn = function(index) {
  return index !== void 0 && __privateGet(this, _keyMap).get(__privateGet(this, _keyList)[index]) === index;
};
_evict = new WeakSet();
evict_fn = function(free) {
  var _a;
  const head = __privateGet(this, _head);
  const k4 = __privateGet(this, _keyList)[head];
  const v7 = __privateGet(this, _valList)[head];
  if (__privateGet(this, _hasFetchMethod) && __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
    v7.__abortController.abort(new Error("evicted"));
  } else if (__privateGet(this, _hasDispose) || __privateGet(this, _hasDisposeAfter)) {
    if (__privateGet(this, _hasDispose)) {
      (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v7, k4, "evict");
    }
    if (__privateGet(this, _hasDisposeAfter)) {
      __privateGet(this, _disposed)?.push([v7, k4, "evict"]);
    }
  }
  __privateGet(this, _removeItemSize).call(this, head);
  if (free) {
    __privateGet(this, _keyList)[head] = void 0;
    __privateGet(this, _valList)[head] = void 0;
    __privateGet(this, _free).push(head);
  }
  if (__privateGet(this, _size) === 1) {
    __privateSet(this, _head, __privateSet(this, _tail, 0));
    __privateGet(this, _free).length = 0;
  } else {
    __privateSet(this, _head, __privateGet(this, _next)[head]);
  }
  __privateGet(this, _keyMap).delete(k4);
  __privateWrapper(this, _size)._--;
  return head;
};
_backgroundFetch = new WeakSet();
backgroundFetch_fn = function(k4, index, options, context) {
  const v7 = index === void 0 ? void 0 : __privateGet(this, _valList)[index];
  if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v7)) {
    return v7;
  }
  const ac = new AC();
  const { signal } = options;
  signal?.addEventListener("abort", () => ac.abort(signal.reason), {
    signal: ac.signal
  });
  const fetchOpts = {
    signal: ac.signal,
    options,
    context
  };
  const cb = (v8, updateCache = false) => {
    const { aborted } = ac.signal;
    const ignoreAbort = options.ignoreFetchAbort && v8 !== void 0;
    if (options.status) {
      if (aborted && !updateCache) {
        options.status.fetchAborted = true;
        options.status.fetchError = ac.signal.reason;
        if (ignoreAbort)
          options.status.fetchAbortIgnored = true;
      } else {
        options.status.fetchResolved = true;
      }
    }
    if (aborted && !ignoreAbort && !updateCache) {
      return fetchFail(ac.signal.reason);
    }
    const bf2 = p7;
    if (__privateGet(this, _valList)[index] === p7) {
      if (v8 === void 0) {
        if (bf2.__staleWhileFetching) {
          __privateGet(this, _valList)[index] = bf2.__staleWhileFetching;
        } else {
          this.delete(k4);
        }
      } else {
        if (options.status)
          options.status.fetchUpdated = true;
        this.set(k4, v8, fetchOpts.options);
      }
    }
    return v8;
  };
  const eb = (er) => {
    if (options.status) {
      options.status.fetchRejected = true;
      options.status.fetchError = er;
    }
    return fetchFail(er);
  };
  const fetchFail = (er) => {
    const { aborted } = ac.signal;
    const allowStaleAborted = aborted && options.allowStaleOnFetchAbort;
    const allowStale = allowStaleAborted || options.allowStaleOnFetchRejection;
    const noDelete = allowStale || options.noDeleteOnFetchRejection;
    const bf2 = p7;
    if (__privateGet(this, _valList)[index] === p7) {
      const del = !noDelete || bf2.__staleWhileFetching === void 0;
      if (del) {
        this.delete(k4);
      } else if (!allowStaleAborted) {
        __privateGet(this, _valList)[index] = bf2.__staleWhileFetching;
      }
    }
    if (allowStale) {
      if (options.status && bf2.__staleWhileFetching !== void 0) {
        options.status.returnedStale = true;
      }
      return bf2.__staleWhileFetching;
    } else if (bf2.__returned === bf2) {
      throw er;
    }
  };
  const pcall = (res, rej) => {
    var _a;
    const fmp = (_a = __privateGet(this, _fetchMethod)) == null ? void 0 : _a.call(this, k4, v7, fetchOpts);
    if (fmp && fmp instanceof Promise) {
      fmp.then((v8) => res(v8), rej);
    }
    ac.signal.addEventListener("abort", () => {
      if (!options.ignoreFetchAbort || options.allowStaleOnFetchAbort) {
        res();
        if (options.allowStaleOnFetchAbort) {
          res = (v8) => cb(v8, true);
        }
      }
    });
  };
  if (options.status)
    options.status.fetchDispatched = true;
  const p7 = new Promise(pcall).then(cb, eb);
  const bf = Object.assign(p7, {
    __abortController: ac,
    __staleWhileFetching: v7,
    __returned: void 0
  });
  if (index === void 0) {
    this.set(k4, bf, { ...fetchOpts.options, status: void 0 });
    index = __privateGet(this, _keyMap).get(k4);
  } else {
    __privateGet(this, _valList)[index] = bf;
  }
  return bf;
};
_isBackgroundFetch = new WeakSet();
isBackgroundFetch_fn = function(p7) {
  if (!__privateGet(this, _hasFetchMethod))
    return false;
  const b5 = p7;
  return !!b5 && b5 instanceof Promise && b5.hasOwnProperty("__staleWhileFetching") && b5.__abortController instanceof AC;
};
_connect = new WeakSet();
connect_fn = function(p7, n9) {
  __privateGet(this, _prev)[n9] = p7;
  __privateGet(this, _next)[p7] = n9;
};
_moveToTail = new WeakSet();
moveToTail_fn = function(index) {
  if (index !== __privateGet(this, _tail)) {
    if (index === __privateGet(this, _head)) {
      __privateSet(this, _head, __privateGet(this, _next)[index]);
    } else {
      __privateMethod(this, _connect, connect_fn).call(this, __privateGet(this, _prev)[index], __privateGet(this, _next)[index]);
    }
    __privateMethod(this, _connect, connect_fn).call(this, __privateGet(this, _tail), index);
    __privateSet(this, _tail, index);
  }
};

// src/CompletionCache.ts
var import_object_hash = __toESM(require_object_hash());
var import_object_sizeof = __toESM(require_indexv2());
var CompletionCache = class {
  constructor() {
    this.logger = rootLogger.child({ component: "CompletionCache" });
    this.options = {
      maxSize: 1 * 1024 * 1024,
      // 1MB
      partiallyAcceptedCacheGeneration: {
        enabled: true,
        perCharacter: {
          lines: 1,
          words: 10,
          max: 30
        },
        perWord: {
          lines: 1,
          max: 20
        },
        perLine: {
          max: 3
        }
      }
    };
    this.cache = new LRUCache({
      maxSize: this.options.maxSize,
      sizeCalculation: import_object_sizeof.default
    });
  }
  has(key) {
    return this.cache.has(this.hash(key));
  }
  set(key, value) {
    for (const entry of this.createCacheEntries(key, value)) {
      this.logger.debug({ entry }, "Setting cache entry");
      this.cache.set(this.hash(entry.key), entry.value);
    }
    this.logger.debug({ size: this.cache.calculatedSize }, "Cache size");
  }
  get(key) {
    return this.cache.get(this.hash(key));
  }
  hash(key) {
    return (0, import_object_hash.default)(key);
  }
  createCacheEntries(key, value) {
    const list = [{ key, value }];
    if (this.options.partiallyAcceptedCacheGeneration.enabled) {
      const entries = value.choices.map((choice) => {
        return this.calculatePartiallyAcceptedPositions(choice.text).map((position) => {
          return {
            prefix: choice.text.slice(0, position),
            suffix: choice.text.slice(position),
            choiceIndex: choice.index
          };
        });
      }).flat().reduce((grouped, entry) => {
        grouped[entry.prefix] = grouped[entry.prefix] || [];
        grouped[entry.prefix].push({ suffix: entry.suffix, choiceIndex: entry.choiceIndex });
        return grouped;
      }, {});
      for (const prefix in entries) {
        const cacheKey = {
          ...key,
          text: key.text.slice(0, key.position) + prefix + key.text.slice(key.position),
          position: key.position + prefix.length
        };
        const cacheValue = {
          ...value,
          choices: entries[prefix].map((choice) => {
            return {
              index: choice.choiceIndex,
              text: choice.suffix
            };
          })
        };
        list.push({
          key: cacheKey,
          value: cacheValue
        });
      }
    }
    return list;
  }
  calculatePartiallyAcceptedPositions(completion) {
    const positions = [];
    const option = this.options.partiallyAcceptedCacheGeneration;
    const lines = splitLines(completion);
    let index = 0;
    let offset = 0;
    while (index < lines.length - 1 && index < option.perLine.max) {
      offset += lines[index].length;
      positions.push(offset);
      index++;
    }
    const words = lines.slice(0, option.perWord.lines).map(splitWords).flat();
    index = 0;
    offset = 0;
    while (index < words.length && index < option.perWord.max) {
      offset += words[index].length;
      positions.push(offset);
      index++;
    }
    const characters = lines.slice(0, option.perCharacter.lines).map(splitWords).flat().slice(0, option.perCharacter.words).join("");
    offset = 1;
    while (offset < characters.length && offset < option.perCharacter.max) {
      positions.push(offset);
      offset++;
    }
    return positions.filter((v7, i7, arr) => arr.indexOf(v7) === i7).sort((a7, b5) => a7 - b5);
  }
};

// src/postprocess/index.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// src/postprocess/filter.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var logger = rootLogger.child({ component: "Postprocess" });
var applyFilter = (filter2) => {
  return async (response) => {
    response.choices = (await Promise.all(
      response.choices.map(async (choice) => {
        choice.text = await filter2(choice.text);
        return choice;
      })
    )).filter(Boolean);
    return response;
  };
};

// src/postprocess/limitScopeByIndentation.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
function calcIndentLevel(line) {
  return line.match(/^[ \t]*/)?.[0]?.length || 0;
}
function isIndentBlockClosingAllowed(currentIndentLevel, suffixLines) {
  let index = 1;
  while (index < suffixLines.length && isBlank(suffixLines[index])) {
    index++;
  }
  if (index >= suffixLines.length) {
    return true;
  } else {
    const indentLevel = calcIndentLevel(suffixLines[index]);
    return indentLevel < currentIndentLevel;
  }
}
function isOpeningIndentBlock(lines, index) {
  if (index >= lines.length - 1) {
    return false;
  }
  return calcIndentLevel(lines[index]) < calcIndentLevel(lines[index + 1]);
}
var limitScopeByIndentation = (context) => {
  return (input) => {
    const prefix = context.text.slice(0, context.position);
    const suffix = context.text.slice(context.position);
    const prefixLines = splitLines(prefix);
    const suffixLines = splitLines(suffix);
    const inputLines = splitLines(input);
    const currentIndentLevel = calcIndentLevel(prefixLines[prefixLines.length - 1]);
    let index;
    for (index = 1; index < inputLines.length; index++) {
      if (isBlank(inputLines[index])) {
        continue;
      }
      const indentLevel = calcIndentLevel(inputLines[index]);
      if (indentLevel < currentIndentLevel) {
        if (isIndentBlockClosingAllowed(currentIndentLevel, suffixLines) && !isOpeningIndentBlock(inputLines, index)) {
          index++;
        }
        break;
      }
    }
    if (index < inputLines.length) {
      logger.debug({ input, prefix, suffix, scopeEndAt: index }, "Remove content out of scope");
      return inputLines.slice(0, index).join("").trimEnd();
    }
    return input;
  };
};

// src/postprocess/removeOverlapping.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var removeOverlapping = (context) => {
  return (input) => {
    const suffix = context.text.slice(context.position);
    for (let index = Math.max(0, input.length - suffix.length); index < input.length; index++) {
      if (input.slice(index) === suffix.slice(0, input.length - index)) {
        logger.debug({ input, suffix, overlappedAt: index }, "Remove overlapped content");
        return input.slice(0, index);
      }
    }
    return input;
  };
};

// src/postprocess/dropBlank.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var dropBlank = () => {
  return (input) => {
    return isBlank(input) ? null : input;
  };
};

// src/postprocess/index.ts
async function postprocess(request2, response) {
  return new Promise((resolve4) => resolve4(response)).then(applyFilter(limitScopeByIndentation(request2))).then(applyFilter(removeOverlapping(request2))).then(applyFilter(dropBlank()));
}

// src/AnonymousUsageLogger.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();

// package.json
var name2 = "tabby-agent";
var version3 = "0.0.1";

// src/AnonymousUsageLogger.ts
var AnonymousUsageLogger = class {
  constructor() {
    this.anonymousUsageTrackingApi = new CloudApi();
    this.logger = rootLogger.child({ component: "AnonymousUsage" });
    this.systemData = {
      agent: `${name2}, ${version3}`,
      browser: navigator?.userAgent || "browser" ,
      node: void 0 
    };
    this.dataStore = null;
  }
  static async create(options) {
    const logger2 = new AnonymousUsageLogger();
    logger2.dataStore = options.dataStore || dataStore;
    await logger2.checkAnonymousId();
    return logger2;
  }
  async checkAnonymousId() {
    if (this.dataStore) {
      try {
        await this.dataStore.load();
      } catch (error) {
        this.logger.debug({ error }, "Error when loading anonymousId");
      }
      if (typeof this.dataStore.data["anonymousId"] === "string") {
        this.anonymousId = this.dataStore.data["anonymousId"];
      } else {
        this.anonymousId = v4_default();
        this.dataStore.data["anonymousId"] = this.anonymousId;
        try {
          await this.dataStore.save();
        } catch (error) {
          this.logger.debug({ error }, "Error when saving anonymousId");
        }
      }
    } else {
      this.anonymousId = v4_default();
    }
  }
  async event(event, data) {
    if (this.disabled) {
      return;
    }
    await this.anonymousUsageTrackingApi.api.usage({
      distinctId: this.anonymousId,
      event,
      properties: {
        ...this.systemData,
        ...data
      }
    }).catch((error) => {
      this.logger.error({ error }, "Error when sending anonymous usage data");
    });
  }
};

// src/TabbyAgent.ts
var _TabbyAgent = class extends EventEmitter {
  constructor() {
    super();
    this.logger = rootLogger.child({ component: "TabbyAgent" });
    this.config = defaultAgentConfig;
    this.status = "notInitialized";
    this.dataStore = null;
    this.completionCache = new CompletionCache();
    // 30s
    this.tryingConnectTimer = null;
    this.tryingConnectTimer = setInterval(async () => {
      if (this.status === "disconnected") {
        this.logger.debug("Trying to connect...");
        await this.healthCheck();
      }
    }, _TabbyAgent.tryConnectInterval);
  }
  static async create(options) {
    const agent = new _TabbyAgent();
    agent.dataStore = options?.dataStore;
    agent.anonymousUsageLogger = await AnonymousUsageLogger.create({ dataStore: options?.dataStore });
    return agent;
  }
  async applyConfig() {
    allLoggers.forEach((logger2) => logger2.level = this.config.logs.level);
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;
    if (this.config.server.endpoint !== this.auth?.endpoint) {
      this.auth = await Auth.create({ endpoint: this.config.server.endpoint, dataStore: this.dataStore });
      this.auth.on("updated", this.setupApi.bind(this));
    }
    await this.setupApi();
  }
  async setupApi() {
    this.api = new TabbyApi({
      BASE: this.config.server.endpoint.replace(/\/+$/, ""),
      // remove trailing slash
      TOKEN: this.auth?.token
    });
    await this.healthCheck();
  }
  changeStatus(status) {
    if (this.status != status) {
      this.status = status;
      const event = { event: "statusChanged", status };
      this.logger.debug({ event }, "Status changed");
      super.emit("statusChanged", event);
    }
  }
  callApi(api, request2) {
    this.logger.debug({ api: api.name, request: request2 }, "API request");
    const promise = api.call(this.api.v1, request2);
    return cancelable(
      promise.then((response) => {
        this.logger.debug({ api: api.name, response }, "API response");
        this.changeStatus("ready");
        return response;
      }).catch((error) => {
        if (!!error.isCancelled) {
          this.logger.debug({ api: api.name, error }, "API request canceled");
        } else if (error.name === "ApiError" && [401, 403, 405].indexOf(error.status) !== -1) {
          this.logger.debug({ api: api.name, error }, "API unauthorized");
          this.changeStatus("unauthorized");
        } else if (error.name === "ApiError") {
          this.logger.error({ api: api.name, error }, "API error");
          this.changeStatus("disconnected");
        } else {
          this.logger.error({ api: api.name, error }, "API request failed with unknown error");
          this.changeStatus("disconnected");
        }
        throw error;
      }),
      () => {
        promise.cancel();
      }
    );
  }
  healthCheck() {
    return this.callApi(this.api.v1.health, {}).catch(() => {
    });
  }
  createSegments(request2) {
    const maxPrefixLines = request2.maxPrefixLines;
    const maxSuffixLines = request2.maxSuffixLines;
    const prefix = request2.text.slice(0, request2.position);
    const prefixLines = splitLines(prefix);
    const suffix = request2.text.slice(request2.position);
    const suffixLines = splitLines(suffix);
    return {
      prefix: prefixLines.slice(Math.max(prefixLines.length - maxPrefixLines, 0)).join(""),
      suffix: suffixLines.slice(0, maxSuffixLines).join("")
    };
  }
  async initialize(options) {
    if (options.client) {
      allLoggers.forEach((logger2) => logger2.setBindings?.({ client: options.client }));
    }
    if (options.config) {
      this.config = (0, import_deepmerge.default)(this.config, options.config);
    }
    await this.applyConfig();
    if (this.status === "unauthorized") {
      const event = { event: "authRequired", server: this.config.server };
      super.emit("authRequired", event);
    }
    await this.anonymousUsageLogger.event("AgentInitialized", {
      client: options.client
    });
    this.logger.debug({ options }, "Initialized");
    return this.status !== "notInitialized";
  }
  async updateConfig(config2) {
    const mergedConfig = (0, import_deepmerge.default)(this.config, config2);
    if (!(0, import_deep_equal.default)(this.config, mergedConfig)) {
      const serverUpdated = !(0, import_deep_equal.default)(this.config.server, mergedConfig.server);
      this.config = mergedConfig;
      await this.applyConfig();
      const event = { event: "configUpdated", config: this.config };
      this.logger.debug({ event }, "Config updated");
      super.emit("configUpdated", event);
      if (serverUpdated && this.status === "unauthorized") {
        const event2 = { event: "authRequired", server: this.config.server };
        super.emit("authRequired", event2);
      }
    }
    return true;
  }
  getConfig() {
    return this.config;
  }
  getStatus() {
    return this.status;
  }
  requestAuthUrl() {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {
      });
    }
    return new CancelablePromise(async (resolve4, reject, onCancel) => {
      let request2;
      onCancel(() => {
        request2?.cancel();
      });
      await this.healthCheck();
      if (onCancel.isCancelled)
        return;
      if (this.status === "unauthorized") {
        request2 = this.auth.requestAuthUrl();
        resolve4(request2);
      }
      resolve4(null);
    });
  }
  waitForAuthToken(code) {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {
      });
    }
    return this.auth.pollingToken(code);
  }
  getCompletions(request2) {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {
      });
    }
    if (this.completionCache.has(request2)) {
      this.logger.debug({ request: request2 }, "Completion cache hit");
      return new CancelablePromise((resolve4) => {
        resolve4(this.completionCache.get(request2));
      });
    }
    const segments = this.createSegments(request2);
    if (isBlank(segments.prefix)) {
      this.logger.debug("Segment prefix is blank, returning empty completion response");
      return new CancelablePromise((resolve4) => {
        resolve4({
          id: "agent-" + v4_default(),
          choices: []
        });
      });
    }
    const promise = this.callApi(this.api.v1.completion, {
      language: request2.language,
      segments,
      user: this.auth?.user
    });
    return cancelable(
      promise.then((response) => {
        this.completionCache.set(request2, response);
        return response;
      }).then((response) => {
        return postprocess(request2, response);
      }),
      () => {
        promise.cancel();
      }
    );
  }
  postEvent(request2) {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {
      });
    }
    return this.callApi(this.api.v1.event, request2);
  }
};
var TabbyAgent = _TabbyAgent;
TabbyAgent.tryConnectInterval = 1e3 * 30;

// src/Agent.ts
init_global();
init_dirname();
init_filename();
init_buffer2();
init_process2();
var agentEventNames = ["statusChanged", "configUpdated", "authRequired"];
/*! Bundled license information:

@jspm/core/nodelibs/browser/buffer.js:
  (*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> *)

@jspm/core/nodelibs/browser/chunk-44e51b61.js:
  (*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> *)

@jspm/core/nodelibs/browser/assert.js:
  (*!
   * The buffer module from node.js, for the browser.
   *
   * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
   * @license  MIT
   *)

ieee754/index.js:
  (*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> *)

buffer/index.js:
  (*!
   * The buffer module from node.js, for the browser.
   *
   * @author   Feross Aboukhadijeh <https://feross.org>
   * @license  MIT
   *)
*/

export { CancelablePromise, TabbyAgent, agentEventNames };
//# sourceMappingURL=out.js.map
//# sourceMappingURL=index.mjs.map
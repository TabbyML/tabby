var Tabby = (() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a2, b) => (typeof require !== "undefined" ? require : a2)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined")
      return require.apply(this, arguments);
    throw new Error('Dynamic require of "' + x + '" is not supported');
  });
  var __esm = (fn, res) => function __init() {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  };
  var __commonJS = (cb, mod) => function __require2() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };
  var __export = (target, all3) => {
    for (var name2 in all3)
      __defProp(target, name2, { get: all3[name2], enumerable: true });
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
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
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
  function unimplemented(name2) {
    throw new Error("Node.js process " + name2 + " is not supported by JSPM core outside of Node.js");
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
      for (var i2 = 1; i2 < arguments.length; i2++)
        args[i2 - 1] = arguments[i2];
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
  function _linkedBinding(name2) {
    unimplemented("_linkedBinding");
  }
  function dlopen(name2) {
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
  function listeners(name2) {
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
      emitWarning = function(message, type) {
        console.warn((type ? type + ": " : "") + message);
      };
      binding = function(name2) {
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
    for (var i2 = 0, len = code.length; i2 < len; ++i2) {
      lookup[i2] = code[i2];
      revLookup[code.charCodeAt(i2)] = i2;
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
      var i3;
      for (i3 = 0; i3 < len2; i3 += 4) {
        tmp = revLookup[b64.charCodeAt(i3)] << 18 | revLookup[b64.charCodeAt(i3 + 1)] << 12 | revLookup[b64.charCodeAt(i3 + 2)] << 6 | revLookup[b64.charCodeAt(i3 + 3)];
        arr[curByte++] = tmp >> 16 & 255;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 2) {
        tmp = revLookup[b64.charCodeAt(i3)] << 2 | revLookup[b64.charCodeAt(i3 + 1)] >> 4;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 1) {
        tmp = revLookup[b64.charCodeAt(i3)] << 10 | revLookup[b64.charCodeAt(i3 + 1)] << 4 | revLookup[b64.charCodeAt(i3 + 2)] >> 2;
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
      for (var i3 = start; i3 < end; i3 += 3) {
        tmp = (uint8[i3] << 16 & 16711680) + (uint8[i3 + 1] << 8 & 65280) + (uint8[i3 + 2] & 255);
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
      for (var i3 = 0, len22 = len2 - extraBytes; i3 < len22; i3 += maxChunkLength) {
        parts.push(encodeChunk(uint8, i3, i3 + maxChunkLength > len22 ? len22 : i3 + maxChunkLength));
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
    exports$2.read = function(buffer, offset, isLE, mLen, nBytes) {
      var e2, m;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var nBits = -7;
      var i2 = isLE ? nBytes - 1 : 0;
      var d = isLE ? -1 : 1;
      var s2 = buffer[offset + i2];
      i2 += d;
      e2 = s2 & (1 << -nBits) - 1;
      s2 >>= -nBits;
      nBits += eLen;
      for (; nBits > 0; e2 = e2 * 256 + buffer[offset + i2], i2 += d, nBits -= 8) {
      }
      m = e2 & (1 << -nBits) - 1;
      e2 >>= -nBits;
      nBits += mLen;
      for (; nBits > 0; m = m * 256 + buffer[offset + i2], i2 += d, nBits -= 8) {
      }
      if (e2 === 0) {
        e2 = 1 - eBias;
      } else if (e2 === eMax) {
        return m ? NaN : (s2 ? -1 : 1) * Infinity;
      } else {
        m = m + Math.pow(2, mLen);
        e2 = e2 - eBias;
      }
      return (s2 ? -1 : 1) * m * Math.pow(2, e2 - mLen);
    };
    exports$2.write = function(buffer, value, offset, isLE, mLen, nBytes) {
      var e2, m, c2;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
      var i2 = isLE ? 0 : nBytes - 1;
      var d = isLE ? 1 : -1;
      var s2 = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
      value = Math.abs(value);
      if (isNaN(value) || value === Infinity) {
        m = isNaN(value) ? 1 : 0;
        e2 = eMax;
      } else {
        e2 = Math.floor(Math.log(value) / Math.LN2);
        if (value * (c2 = Math.pow(2, -e2)) < 1) {
          e2--;
          c2 *= 2;
        }
        if (e2 + eBias >= 1) {
          value += rt / c2;
        } else {
          value += rt * Math.pow(2, 1 - eBias);
        }
        if (value * c2 >= 2) {
          e2++;
          c2 /= 2;
        }
        if (e2 + eBias >= eMax) {
          m = 0;
          e2 = eMax;
        } else if (e2 + eBias >= 1) {
          m = (value * c2 - 1) * Math.pow(2, mLen);
          e2 = e2 + eBias;
        } else {
          m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
          e2 = 0;
        }
      }
      for (; mLen >= 8; buffer[offset + i2] = m & 255, i2 += d, m /= 256, mLen -= 8) {
      }
      e2 = e2 << mLen | m;
      eLen += mLen;
      for (; eLen > 0; buffer[offset + i2] = e2 & 255, i2 += d, e2 /= 256, eLen -= 8) {
      }
      buffer[offset + i2 - d] |= s2 * 128;
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
      } catch (e2) {
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
      const b = fromObject(value);
      if (b)
        return b;
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
      for (let i2 = 0; i2 < length; i2 += 1) {
        buf[i2] = array[i2] & 255;
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
    Buffer3.isBuffer = function isBuffer2(b) {
      return b != null && b._isBuffer === true && b !== Buffer3.prototype;
    };
    Buffer3.compare = function compare(a2, b) {
      if (isInstance(a2, Uint8Array))
        a2 = Buffer3.from(a2, a2.offset, a2.byteLength);
      if (isInstance(b, Uint8Array))
        b = Buffer3.from(b, b.offset, b.byteLength);
      if (!Buffer3.isBuffer(a2) || !Buffer3.isBuffer(b)) {
        throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');
      }
      if (a2 === b)
        return 0;
      let x = a2.length;
      let y2 = b.length;
      for (let i2 = 0, len = Math.min(x, y2); i2 < len; ++i2) {
        if (a2[i2] !== b[i2]) {
          x = a2[i2];
          y2 = b[i2];
          break;
        }
      }
      if (x < y2)
        return -1;
      if (y2 < x)
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
      let i2;
      if (length === void 0) {
        length = 0;
        for (i2 = 0; i2 < list.length; ++i2) {
          length += list[i2].length;
        }
      }
      const buffer = Buffer3.allocUnsafe(length);
      let pos = 0;
      for (i2 = 0; i2 < list.length; ++i2) {
        let buf = list[i2];
        if (isInstance(buf, Uint8Array)) {
          if (pos + buf.length > buffer.length) {
            if (!Buffer3.isBuffer(buf))
              buf = Buffer3.from(buf);
            buf.copy(buffer, pos);
          } else {
            Uint8Array.prototype.set.call(buffer, buf, pos);
          }
        } else if (!Buffer3.isBuffer(buf)) {
          throw new TypeError('"list" argument must be an Array of Buffers');
        } else {
          buf.copy(buffer, pos);
        }
        pos += buf.length;
      }
      return buffer;
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
    function swap(b, n2, m) {
      const i2 = b[n2];
      b[n2] = b[m];
      b[m] = i2;
    }
    Buffer3.prototype.swap16 = function swap16() {
      const len = this.length;
      if (len % 2 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 16-bits");
      }
      for (let i2 = 0; i2 < len; i2 += 2) {
        swap(this, i2, i2 + 1);
      }
      return this;
    };
    Buffer3.prototype.swap32 = function swap32() {
      const len = this.length;
      if (len % 4 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 32-bits");
      }
      for (let i2 = 0; i2 < len; i2 += 4) {
        swap(this, i2, i2 + 3);
        swap(this, i2 + 1, i2 + 2);
      }
      return this;
    };
    Buffer3.prototype.swap64 = function swap64() {
      const len = this.length;
      if (len % 8 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 64-bits");
      }
      for (let i2 = 0; i2 < len; i2 += 8) {
        swap(this, i2, i2 + 7);
        swap(this, i2 + 1, i2 + 6);
        swap(this, i2 + 2, i2 + 5);
        swap(this, i2 + 3, i2 + 4);
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
    Buffer3.prototype.equals = function equals(b) {
      if (!Buffer3.isBuffer(b))
        throw new TypeError("Argument must be a Buffer");
      if (this === b)
        return true;
      return Buffer3.compare(this, b) === 0;
    };
    Buffer3.prototype.inspect = function inspect() {
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
      let x = thisEnd - thisStart;
      let y2 = end - start;
      const len = Math.min(x, y2);
      const thisCopy = this.slice(thisStart, thisEnd);
      const targetCopy = target.slice(start, end);
      for (let i2 = 0; i2 < len; ++i2) {
        if (thisCopy[i2] !== targetCopy[i2]) {
          x = thisCopy[i2];
          y2 = targetCopy[i2];
          break;
        }
      }
      if (x < y2)
        return -1;
      if (y2 < x)
        return 1;
      return 0;
    };
    function bidirectionalIndexOf(buffer, val, byteOffset, encoding, dir) {
      if (buffer.length === 0)
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
        byteOffset = dir ? 0 : buffer.length - 1;
      }
      if (byteOffset < 0)
        byteOffset = buffer.length + byteOffset;
      if (byteOffset >= buffer.length) {
        if (dir)
          return -1;
        else
          byteOffset = buffer.length - 1;
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
        return arrayIndexOf(buffer, val, byteOffset, encoding, dir);
      } else if (typeof val === "number") {
        val = val & 255;
        if (typeof Uint8Array.prototype.indexOf === "function") {
          if (dir) {
            return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset);
          } else {
            return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset);
          }
        }
        return arrayIndexOf(buffer, [val], byteOffset, encoding, dir);
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
      function read(buf, i3) {
        if (indexSize === 1) {
          return buf[i3];
        } else {
          return buf.readUInt16BE(i3 * indexSize);
        }
      }
      let i2;
      if (dir) {
        let foundIndex = -1;
        for (i2 = byteOffset; i2 < arrLength; i2++) {
          if (read(arr, i2) === read(val, foundIndex === -1 ? 0 : i2 - foundIndex)) {
            if (foundIndex === -1)
              foundIndex = i2;
            if (i2 - foundIndex + 1 === valLength)
              return foundIndex * indexSize;
          } else {
            if (foundIndex !== -1)
              i2 -= i2 - foundIndex;
            foundIndex = -1;
          }
        }
      } else {
        if (byteOffset + valLength > arrLength)
          byteOffset = arrLength - valLength;
        for (i2 = byteOffset; i2 >= 0; i2--) {
          let found = true;
          for (let j = 0; j < valLength; j++) {
            if (read(arr, i2 + j) !== read(val, j)) {
              found = false;
              break;
            }
          }
          if (found)
            return i2;
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
      let i2;
      for (i2 = 0; i2 < length; ++i2) {
        const parsed = parseInt(string.substr(i2 * 2, 2), 16);
        if (numberIsNaN(parsed))
          return i2;
        buf[offset + i2] = parsed;
      }
      return i2;
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
    Buffer3.prototype.write = function write(string, offset, length, encoding) {
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
      let i2 = start;
      while (i2 < end) {
        const firstByte = buf[i2];
        let codePoint = null;
        let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
        if (i2 + bytesPerSequence <= end) {
          let secondByte, thirdByte, fourthByte, tempCodePoint;
          switch (bytesPerSequence) {
            case 1:
              if (firstByte < 128) {
                codePoint = firstByte;
              }
              break;
            case 2:
              secondByte = buf[i2 + 1];
              if ((secondByte & 192) === 128) {
                tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
                if (tempCodePoint > 127) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 3:
              secondByte = buf[i2 + 1];
              thirdByte = buf[i2 + 2];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
                if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 4:
              secondByte = buf[i2 + 1];
              thirdByte = buf[i2 + 2];
              fourthByte = buf[i2 + 3];
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
        i2 += bytesPerSequence;
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
      let i2 = 0;
      while (i2 < len) {
        res += String.fromCharCode.apply(String, codePoints.slice(i2, i2 += MAX_ARGUMENTS_LENGTH));
      }
      return res;
    }
    function asciiSlice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i2 = start; i2 < end; ++i2) {
        ret += String.fromCharCode(buf[i2] & 127);
      }
      return ret;
    }
    function latin1Slice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i2 = start; i2 < end; ++i2) {
        ret += String.fromCharCode(buf[i2]);
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
      for (let i2 = start; i2 < end; ++i2) {
        out += hexSliceLookupTable[buf[i2]];
      }
      return out;
    }
    function utf16leSlice(buf, start, end) {
      const bytes = buf.slice(start, end);
      let res = "";
      for (let i2 = 0; i2 < bytes.length - 1; i2 += 2) {
        res += String.fromCharCode(bytes[i2] + bytes[i2 + 1] * 256);
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
      let i2 = 0;
      while (++i2 < byteLength2 && (mul *= 256)) {
        val += this[offset + i2] * mul;
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
      let i2 = 0;
      while (++i2 < byteLength2 && (mul *= 256)) {
        val += this[offset + i2] * mul;
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
      let i2 = byteLength2;
      let mul = 1;
      let val = this[offset + --i2];
      while (i2 > 0 && (mul *= 256)) {
        val += this[offset + --i2] * mul;
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
      let i2 = 0;
      this[offset] = value & 255;
      while (++i2 < byteLength2 && (mul *= 256)) {
        this[offset + i2] = value / mul & 255;
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
      let i2 = byteLength2 - 1;
      let mul = 1;
      this[offset + i2] = value & 255;
      while (--i2 >= 0 && (mul *= 256)) {
        this[offset + i2] = value / mul & 255;
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
      let i2 = 0;
      let mul = 1;
      let sub = 0;
      this[offset] = value & 255;
      while (++i2 < byteLength2 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i2 - 1] !== 0) {
          sub = 1;
        }
        this[offset + i2] = (value / mul >> 0) - sub & 255;
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
      let i2 = byteLength2 - 1;
      let mul = 1;
      let sub = 0;
      this[offset + i2] = value & 255;
      while (--i2 >= 0 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i2 + 1] !== 0) {
          sub = 1;
        }
        this[offset + i2] = (value / mul >> 0) - sub & 255;
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
      let i2;
      if (typeof val === "number") {
        for (i2 = start; i2 < end; ++i2) {
          this[i2] = val;
        }
      } else {
        const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
        const len = bytes.length;
        if (len === 0) {
          throw new TypeError('The value "' + val + '" is invalid for argument "value"');
        }
        for (i2 = 0; i2 < end - start; ++i2) {
          this[i2 + start] = bytes[i2 % len];
        }
      }
      return this;
    };
    const errors = {};
    function E(sym, getMessage, Base) {
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
    E("ERR_BUFFER_OUT_OF_BOUNDS", function(name2) {
      if (name2) {
        return `${name2} is outside of buffer bounds`;
      }
      return "Attempt to access memory outside buffer bounds";
    }, RangeError);
    E("ERR_INVALID_ARG_TYPE", function(name2, actual) {
      return `The "${name2}" argument must be of type number. Received type ${typeof actual}`;
    }, TypeError);
    E("ERR_OUT_OF_RANGE", function(str, range, input) {
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
      let i2 = val.length;
      const start = val[0] === "-" ? 1 : 0;
      for (; i2 >= start + 4; i2 -= 3) {
        res = `_${val.slice(i2 - 3, i2)}${res}`;
      }
      return `${val.slice(0, i2)}${res}`;
    }
    function checkBounds(buf, offset, byteLength2) {
      validateNumber(offset, "offset");
      if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
        boundsError(offset, buf.length - (byteLength2 + 1));
      }
    }
    function checkIntBI(value, min, max, buf, offset, byteLength2) {
      if (value > max || value < min) {
        const n2 = typeof min === "bigint" ? "n" : "";
        let range;
        if (byteLength2 > 3) {
          if (min === 0 || min === BigInt(0)) {
            range = `>= 0${n2} and < 2${n2} ** ${(byteLength2 + 1) * 8}${n2}`;
          } else {
            range = `>= -(2${n2} ** ${(byteLength2 + 1) * 8 - 1}${n2}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n2}`;
          }
        } else {
          range = `>= ${min}${n2} and <= ${max}${n2}`;
        }
        throw new errors.ERR_OUT_OF_RANGE("value", range, value);
      }
      checkBounds(buf, offset, byteLength2);
    }
    function validateNumber(value, name2) {
      if (typeof value !== "number") {
        throw new errors.ERR_INVALID_ARG_TYPE(name2, "number", value);
      }
    }
    function boundsError(value, length, type) {
      if (Math.floor(value) !== value) {
        validateNumber(value, type);
        throw new errors.ERR_OUT_OF_RANGE(type || "offset", "an integer", value);
      }
      if (length < 0) {
        throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
      }
      throw new errors.ERR_OUT_OF_RANGE(type || "offset", `>= ${type ? 1 : 0} and <= ${length}`, value);
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
      for (let i2 = 0; i2 < length; ++i2) {
        codePoint = string.charCodeAt(i2);
        if (codePoint > 55295 && codePoint < 57344) {
          if (!leadSurrogate) {
            if (codePoint > 56319) {
              if ((units -= 3) > -1)
                bytes.push(239, 191, 189);
              continue;
            } else if (i2 + 1 === length) {
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
      for (let i2 = 0; i2 < str.length; ++i2) {
        byteArray.push(str.charCodeAt(i2) & 255);
      }
      return byteArray;
    }
    function utf16leToBytes(str, units) {
      let c2, hi, lo;
      const byteArray = [];
      for (let i2 = 0; i2 < str.length; ++i2) {
        if ((units -= 2) < 0)
          break;
        c2 = str.charCodeAt(i2);
        hi = c2 >> 8;
        lo = c2 % 256;
        byteArray.push(lo);
        byteArray.push(hi);
      }
      return byteArray;
    }
    function base64ToBytes(str) {
      return base642.toByteArray(base64clean(str));
    }
    function blitBuffer(src, dst, offset, length) {
      let i2;
      for (i2 = 0; i2 < length; ++i2) {
        if (i2 + offset >= dst.length || i2 >= src.length)
          break;
        dst[i2 + offset] = src[i2];
      }
      return i2;
    }
    function isInstance(obj, type) {
      return obj instanceof type || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type.name;
    }
    function numberIsNaN(obj) {
      return obj !== obj;
    }
    const hexSliceLookupTable = function() {
      const alphabet = "0123456789abcdef";
      const table = new Array(256);
      for (let i2 = 0; i2 < 16; ++i2) {
        const i16 = i2 * 16;
        for (let j = 0; j < 16; ++j) {
          table[i16 + j] = alphabet[i2] + alphabet[j];
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
  var exports$3, _dewExec$2, exports$2, _dewExec$1, exports$1, _dewExec, exports, Buffer2, INSPECT_MAX_BYTES, kMaxLength;
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
      INSPECT_MAX_BYTES = exports.INSPECT_MAX_BYTES;
      kMaxLength = exports.kMaxLength;
    }
  });

  // node_modules/esbuild-plugin-polyfill-node/polyfills/buffer.js
  var init_buffer2 = __esm({
    "node_modules/esbuild-plugin-polyfill-node/polyfills/buffer.js"() {
      init_buffer();
    }
  });

  // node_modules/object-hash/dist/object_hash.js
  var require_object_hash = __commonJS({
    "node_modules/object-hash/dist/object_hash.js"(exports2, module) {
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      !function(e2) {
        var t2;
        "object" == typeof exports2 ? module.exports = e2() : "function" == typeof define && define.amd ? define(e2) : ("undefined" != typeof window ? t2 = window : "undefined" != typeof global ? t2 = global : "undefined" != typeof self && (t2 = self), t2.objectHash = e2());
      }(function() {
        return function r2(o2, i2, u2) {
          function s2(n2, e3) {
            if (!i2[n2]) {
              if (!o2[n2]) {
                var t2 = "function" == typeof __require && __require;
                if (!e3 && t2)
                  return t2(n2, true);
                if (a2)
                  return a2(n2, true);
                throw new Error("Cannot find module '" + n2 + "'");
              }
              e3 = i2[n2] = { exports: {} };
              o2[n2][0].call(e3.exports, function(e4) {
                var t3 = o2[n2][1][e4];
                return s2(t3 || e4);
              }, e3, e3.exports, r2, o2, i2, u2);
            }
            return i2[n2].exports;
          }
          for (var a2 = "function" == typeof __require && __require, e2 = 0; e2 < u2.length; e2++)
            s2(u2[e2]);
          return s2;
        }({ 1: [function(w, b, m) {
          !function(e2, n2, s2, c2, d, h2, p2, g, y2) {
            "use strict";
            var r2 = w("crypto");
            function t2(e3, t3) {
              t3 = u2(e3, t3);
              var n3;
              return void 0 === (n3 = "passthrough" !== t3.algorithm ? r2.createHash(t3.algorithm) : new l2()).write && (n3.write = n3.update, n3.end = n3.update), f2(t3, n3).dispatch(e3), n3.update || n3.end(""), n3.digest ? n3.digest("buffer" === t3.encoding ? void 0 : t3.encoding) : (e3 = n3.read(), "buffer" !== t3.encoding ? e3.toString(t3.encoding) : e3);
            }
            (m = b.exports = t2).sha1 = function(e3) {
              return t2(e3);
            }, m.keys = function(e3) {
              return t2(e3, { excludeValues: true, algorithm: "sha1", encoding: "hex" });
            }, m.MD5 = function(e3) {
              return t2(e3, { algorithm: "md5", encoding: "hex" });
            }, m.keysMD5 = function(e3) {
              return t2(e3, { algorithm: "md5", encoding: "hex", excludeValues: true });
            };
            var o2 = r2.getHashes ? r2.getHashes().slice() : ["sha1", "md5"], i2 = (o2.push("passthrough"), ["buffer", "hex", "binary", "base64"]);
            function u2(e3, t3) {
              var n3 = {};
              if (n3.algorithm = (t3 = t3 || {}).algorithm || "sha1", n3.encoding = t3.encoding || "hex", n3.excludeValues = !!t3.excludeValues, n3.algorithm = n3.algorithm.toLowerCase(), n3.encoding = n3.encoding.toLowerCase(), n3.ignoreUnknown = true === t3.ignoreUnknown, n3.respectType = false !== t3.respectType, n3.respectFunctionNames = false !== t3.respectFunctionNames, n3.respectFunctionProperties = false !== t3.respectFunctionProperties, n3.unorderedArrays = true === t3.unorderedArrays, n3.unorderedSets = false !== t3.unorderedSets, n3.unorderedObjects = false !== t3.unorderedObjects, n3.replacer = t3.replacer || void 0, n3.excludeKeys = t3.excludeKeys || void 0, void 0 === e3)
                throw new Error("Object argument required.");
              for (var r3 = 0; r3 < o2.length; ++r3)
                o2[r3].toLowerCase() === n3.algorithm.toLowerCase() && (n3.algorithm = o2[r3]);
              if (-1 === o2.indexOf(n3.algorithm))
                throw new Error('Algorithm "' + n3.algorithm + '"  not supported. supported values: ' + o2.join(", "));
              if (-1 === i2.indexOf(n3.encoding) && "passthrough" !== n3.algorithm)
                throw new Error('Encoding "' + n3.encoding + '"  not supported. supported values: ' + i2.join(", "));
              return n3;
            }
            function a2(e3) {
              if ("function" == typeof e3)
                return null != /^function\s+\w*\s*\(\s*\)\s*{\s+\[native code\]\s+}$/i.exec(Function.prototype.toString.call(e3));
            }
            function f2(o3, t3, i3) {
              i3 = i3 || [];
              function u3(e3) {
                return t3.update ? t3.update(e3, "utf8") : t3.write(e3, "utf8");
              }
              return { dispatch: function(e3) {
                return this["_" + (null === (e3 = o3.replacer ? o3.replacer(e3) : e3) ? "null" : typeof e3)](e3);
              }, _object: function(t4) {
                var n3, e3 = Object.prototype.toString.call(t4), r3 = /\[object (.*)\]/i.exec(e3);
                r3 = (r3 = r3 ? r3[1] : "unknown:[" + e3 + "]").toLowerCase();
                if (0 <= (e3 = i3.indexOf(t4)))
                  return this.dispatch("[CIRCULAR:" + e3 + "]");
                if (i3.push(t4), void 0 !== s2 && s2.isBuffer && s2.isBuffer(t4))
                  return u3("buffer:"), u3(t4);
                if ("object" === r3 || "function" === r3 || "asyncfunction" === r3)
                  return e3 = Object.keys(t4), o3.unorderedObjects && (e3 = e3.sort()), false === o3.respectType || a2(t4) || e3.splice(0, 0, "prototype", "__proto__", "constructor"), o3.excludeKeys && (e3 = e3.filter(function(e4) {
                    return !o3.excludeKeys(e4);
                  })), u3("object:" + e3.length + ":"), n3 = this, e3.forEach(function(e4) {
                    n3.dispatch(e4), u3(":"), o3.excludeValues || n3.dispatch(t4[e4]), u3(",");
                  });
                if (!this["_" + r3]) {
                  if (o3.ignoreUnknown)
                    return u3("[" + r3 + "]");
                  throw new Error('Unknown object type "' + r3 + '"');
                }
                this["_" + r3](t4);
              }, _array: function(e3, t4) {
                t4 = void 0 !== t4 ? t4 : false !== o3.unorderedArrays;
                var n3 = this;
                if (u3("array:" + e3.length + ":"), !t4 || e3.length <= 1)
                  return e3.forEach(function(e4) {
                    return n3.dispatch(e4);
                  });
                var r3 = [], t4 = e3.map(function(e4) {
                  var t5 = new l2(), n4 = i3.slice();
                  return f2(o3, t5, n4).dispatch(e4), r3 = r3.concat(n4.slice(i3.length)), t5.read().toString();
                });
                return i3 = i3.concat(r3), t4.sort(), this._array(t4, false);
              }, _date: function(e3) {
                return u3("date:" + e3.toJSON());
              }, _symbol: function(e3) {
                return u3("symbol:" + e3.toString());
              }, _error: function(e3) {
                return u3("error:" + e3.toString());
              }, _boolean: function(e3) {
                return u3("bool:" + e3.toString());
              }, _string: function(e3) {
                u3("string:" + e3.length + ":"), u3(e3.toString());
              }, _function: function(e3) {
                u3("fn:"), a2(e3) ? this.dispatch("[native]") : this.dispatch(e3.toString()), false !== o3.respectFunctionNames && this.dispatch("function-name:" + String(e3.name)), o3.respectFunctionProperties && this._object(e3);
              }, _number: function(e3) {
                return u3("number:" + e3.toString());
              }, _xml: function(e3) {
                return u3("xml:" + e3.toString());
              }, _null: function() {
                return u3("Null");
              }, _undefined: function() {
                return u3("Undefined");
              }, _regexp: function(e3) {
                return u3("regex:" + e3.toString());
              }, _uint8array: function(e3) {
                return u3("uint8array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _uint8clampedarray: function(e3) {
                return u3("uint8clampedarray:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _int8array: function(e3) {
                return u3("int8array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _uint16array: function(e3) {
                return u3("uint16array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _int16array: function(e3) {
                return u3("int16array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _uint32array: function(e3) {
                return u3("uint32array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _int32array: function(e3) {
                return u3("int32array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _float32array: function(e3) {
                return u3("float32array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _float64array: function(e3) {
                return u3("float64array:"), this.dispatch(Array.prototype.slice.call(e3));
              }, _arraybuffer: function(e3) {
                return u3("arraybuffer:"), this.dispatch(new Uint8Array(e3));
              }, _url: function(e3) {
                return u3("url:" + e3.toString());
              }, _map: function(e3) {
                u3("map:");
                e3 = Array.from(e3);
                return this._array(e3, false !== o3.unorderedSets);
              }, _set: function(e3) {
                u3("set:");
                e3 = Array.from(e3);
                return this._array(e3, false !== o3.unorderedSets);
              }, _file: function(e3) {
                return u3("file:"), this.dispatch([e3.name, e3.size, e3.type, e3.lastModfied]);
              }, _blob: function() {
                if (o3.ignoreUnknown)
                  return u3("[blob]");
                throw Error('Hashing Blob objects is currently not supported\n(see https://github.com/puleos/object-hash/issues/26)\nUse "options.replacer" or "options.ignoreUnknown"\n');
              }, _domwindow: function() {
                return u3("domwindow");
              }, _bigint: function(e3) {
                return u3("bigint:" + e3.toString());
              }, _process: function() {
                return u3("process");
              }, _timer: function() {
                return u3("timer");
              }, _pipe: function() {
                return u3("pipe");
              }, _tcp: function() {
                return u3("tcp");
              }, _udp: function() {
                return u3("udp");
              }, _tty: function() {
                return u3("tty");
              }, _statwatcher: function() {
                return u3("statwatcher");
              }, _securecontext: function() {
                return u3("securecontext");
              }, _connection: function() {
                return u3("connection");
              }, _zlib: function() {
                return u3("zlib");
              }, _context: function() {
                return u3("context");
              }, _nodescript: function() {
                return u3("nodescript");
              }, _httpparser: function() {
                return u3("httpparser");
              }, _dataview: function() {
                return u3("dataview");
              }, _signal: function() {
                return u3("signal");
              }, _fsevent: function() {
                return u3("fsevent");
              }, _tlswrap: function() {
                return u3("tlswrap");
              } };
            }
            function l2() {
              return { buf: "", write: function(e3) {
                this.buf += e3;
              }, end: function(e3) {
                this.buf += e3;
              }, read: function() {
                return this.buf;
              } };
            }
            m.writeToStream = function(e3, t3, n3) {
              return void 0 === n3 && (n3 = t3, t3 = {}), f2(t3 = u2(e3, t3), n3).dispatch(e3);
            };
          }.call(this, w("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, w("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/fake_9a5aa49d.js", "/");
        }, { buffer: 3, crypto: 5, lYpoI2: 11 }], 2: [function(e2, t2, f2) {
          !function(e3, t3, n2, r2, o2, i2, u2, s2, a2) {
            !function(e4) {
              "use strict";
              var a3 = "undefined" != typeof Uint8Array ? Uint8Array : Array, t4 = "+".charCodeAt(0), n3 = "/".charCodeAt(0), r3 = "0".charCodeAt(0), o3 = "a".charCodeAt(0), i3 = "A".charCodeAt(0), u3 = "-".charCodeAt(0), s3 = "_".charCodeAt(0);
              function f3(e5) {
                e5 = e5.charCodeAt(0);
                return e5 === t4 || e5 === u3 ? 62 : e5 === n3 || e5 === s3 ? 63 : e5 < r3 ? -1 : e5 < r3 + 10 ? e5 - r3 + 26 + 26 : e5 < i3 + 26 ? e5 - i3 : e5 < o3 + 26 ? e5 - o3 + 26 : void 0;
              }
              e4.toByteArray = function(e5) {
                var t5, n4;
                if (0 < e5.length % 4)
                  throw new Error("Invalid string. Length must be a multiple of 4");
                var r4 = e5.length, r4 = "=" === e5.charAt(r4 - 2) ? 2 : "=" === e5.charAt(r4 - 1) ? 1 : 0, o4 = new a3(3 * e5.length / 4 - r4), i4 = 0 < r4 ? e5.length - 4 : e5.length, u4 = 0;
                function s4(e6) {
                  o4[u4++] = e6;
                }
                for (t5 = 0; t5 < i4; t5 += 4, 0)
                  s4((16711680 & (n4 = f3(e5.charAt(t5)) << 18 | f3(e5.charAt(t5 + 1)) << 12 | f3(e5.charAt(t5 + 2)) << 6 | f3(e5.charAt(t5 + 3)))) >> 16), s4((65280 & n4) >> 8), s4(255 & n4);
                return 2 == r4 ? s4(255 & (n4 = f3(e5.charAt(t5)) << 2 | f3(e5.charAt(t5 + 1)) >> 4)) : 1 == r4 && (s4((n4 = f3(e5.charAt(t5)) << 10 | f3(e5.charAt(t5 + 1)) << 4 | f3(e5.charAt(t5 + 2)) >> 2) >> 8 & 255), s4(255 & n4)), o4;
              }, e4.fromByteArray = function(e5) {
                var t5, n4, r4, o4, i4 = e5.length % 3, u4 = "";
                function s4(e6) {
                  return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(e6);
                }
                for (t5 = 0, r4 = e5.length - i4; t5 < r4; t5 += 3)
                  n4 = (e5[t5] << 16) + (e5[t5 + 1] << 8) + e5[t5 + 2], u4 += s4((o4 = n4) >> 18 & 63) + s4(o4 >> 12 & 63) + s4(o4 >> 6 & 63) + s4(63 & o4);
                switch (i4) {
                  case 1:
                    u4 = (u4 += s4((n4 = e5[e5.length - 1]) >> 2)) + s4(n4 << 4 & 63) + "==";
                    break;
                  case 2:
                    u4 = (u4 = (u4 += s4((n4 = (e5[e5.length - 2] << 8) + e5[e5.length - 1]) >> 10)) + s4(n4 >> 4 & 63)) + s4(n4 << 2 & 63) + "=";
                }
                return u4;
              };
            }(void 0 === f2 ? this.base64js = {} : f2);
          }.call(this, e2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/base64-js/lib/b64.js", "/node_modules/gulp-browserify/node_modules/base64-js/lib");
        }, { buffer: 3, lYpoI2: 11 }], 3: [function(O, e2, H) {
          !function(e3, n2, f2, r2, h2, p2, g, y2, w) {
            var a2 = O("base64-js"), i2 = O("ieee754");
            function f2(e4, t3, n3) {
              if (!(this instanceof f2))
                return new f2(e4, t3, n3);
              var r3, o3, i3, u3, s3 = typeof e4;
              if ("base64" === t3 && "string" == s3)
                for (e4 = (u3 = e4).trim ? u3.trim() : u3.replace(/^\s+|\s+$/g, ""); e4.length % 4 != 0; )
                  e4 += "=";
              if ("number" == s3)
                r3 = j(e4);
              else if ("string" == s3)
                r3 = f2.byteLength(e4, t3);
              else {
                if ("object" != s3)
                  throw new Error("First argument needs to be a number, array or string.");
                r3 = j(e4.length);
              }
              if (f2._useTypedArrays ? o3 = f2._augment(new Uint8Array(r3)) : ((o3 = this).length = r3, o3._isBuffer = true), f2._useTypedArrays && "number" == typeof e4.byteLength)
                o3._set(e4);
              else if (C(u3 = e4) || f2.isBuffer(u3) || u3 && "object" == typeof u3 && "number" == typeof u3.length)
                for (i3 = 0; i3 < r3; i3++)
                  f2.isBuffer(e4) ? o3[i3] = e4.readUInt8(i3) : o3[i3] = e4[i3];
              else if ("string" == s3)
                o3.write(e4, 0, t3);
              else if ("number" == s3 && !f2._useTypedArrays && !n3)
                for (i3 = 0; i3 < r3; i3++)
                  o3[i3] = 0;
              return o3;
            }
            function b(e4, t3, n3, r3) {
              return f2._charsWritten = c2(function(e5) {
                for (var t4 = [], n4 = 0; n4 < e5.length; n4++)
                  t4.push(255 & e5.charCodeAt(n4));
                return t4;
              }(t3), e4, n3, r3);
            }
            function m(e4, t3, n3, r3) {
              return f2._charsWritten = c2(function(e5) {
                for (var t4, n4, r4 = [], o3 = 0; o3 < e5.length; o3++)
                  n4 = e5.charCodeAt(o3), t4 = n4 >> 8, n4 = n4 % 256, r4.push(n4), r4.push(t4);
                return r4;
              }(t3), e4, n3, r3);
            }
            function v2(e4, t3, n3) {
              var r3 = "";
              n3 = Math.min(e4.length, n3);
              for (var o3 = t3; o3 < n3; o3++)
                r3 += String.fromCharCode(e4[o3]);
              return r3;
            }
            function o2(e4, t3, n3, r3) {
              r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(null != t3, "missing offset"), d(t3 + 1 < e4.length, "Trying to read beyond buffer length"));
              var o3, r3 = e4.length;
              if (!(r3 <= t3))
                return n3 ? (o3 = e4[t3], t3 + 1 < r3 && (o3 |= e4[t3 + 1] << 8)) : (o3 = e4[t3] << 8, t3 + 1 < r3 && (o3 |= e4[t3 + 1])), o3;
            }
            function u2(e4, t3, n3, r3) {
              r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(null != t3, "missing offset"), d(t3 + 3 < e4.length, "Trying to read beyond buffer length"));
              var o3, r3 = e4.length;
              if (!(r3 <= t3))
                return n3 ? (t3 + 2 < r3 && (o3 = e4[t3 + 2] << 16), t3 + 1 < r3 && (o3 |= e4[t3 + 1] << 8), o3 |= e4[t3], t3 + 3 < r3 && (o3 += e4[t3 + 3] << 24 >>> 0)) : (t3 + 1 < r3 && (o3 = e4[t3 + 1] << 16), t3 + 2 < r3 && (o3 |= e4[t3 + 2] << 8), t3 + 3 < r3 && (o3 |= e4[t3 + 3]), o3 += e4[t3] << 24 >>> 0), o3;
            }
            function _(e4, t3, n3, r3) {
              if (r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(null != t3, "missing offset"), d(t3 + 1 < e4.length, "Trying to read beyond buffer length")), !(e4.length <= t3))
                return r3 = o2(e4, t3, n3, true), 32768 & r3 ? -1 * (65535 - r3 + 1) : r3;
            }
            function E(e4, t3, n3, r3) {
              if (r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(null != t3, "missing offset"), d(t3 + 3 < e4.length, "Trying to read beyond buffer length")), !(e4.length <= t3))
                return r3 = u2(e4, t3, n3, true), 2147483648 & r3 ? -1 * (4294967295 - r3 + 1) : r3;
            }
            function I(e4, t3, n3, r3) {
              return r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(t3 + 3 < e4.length, "Trying to read beyond buffer length")), i2.read(e4, t3, n3, 23, 4);
            }
            function A(e4, t3, n3, r3) {
              return r3 || (d("boolean" == typeof n3, "missing or invalid endian"), d(t3 + 7 < e4.length, "Trying to read beyond buffer length")), i2.read(e4, t3, n3, 52, 8);
            }
            function s2(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 1 < e4.length, "trying to write beyond buffer length"), Y(t3, 65535));
              o3 = e4.length;
              if (!(o3 <= n3))
                for (var i3 = 0, u3 = Math.min(o3 - n3, 2); i3 < u3; i3++)
                  e4[n3 + i3] = (t3 & 255 << 8 * (r3 ? i3 : 1 - i3)) >>> 8 * (r3 ? i3 : 1 - i3);
            }
            function l2(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 3 < e4.length, "trying to write beyond buffer length"), Y(t3, 4294967295));
              o3 = e4.length;
              if (!(o3 <= n3))
                for (var i3 = 0, u3 = Math.min(o3 - n3, 4); i3 < u3; i3++)
                  e4[n3 + i3] = t3 >>> 8 * (r3 ? i3 : 3 - i3) & 255;
            }
            function B(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 1 < e4.length, "Trying to write beyond buffer length"), F(t3, 32767, -32768)), e4.length <= n3 || s2(e4, 0 <= t3 ? t3 : 65535 + t3 + 1, n3, r3, o3);
            }
            function L(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 3 < e4.length, "Trying to write beyond buffer length"), F(t3, 2147483647, -2147483648)), e4.length <= n3 || l2(e4, 0 <= t3 ? t3 : 4294967295 + t3 + 1, n3, r3, o3);
            }
            function U(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 3 < e4.length, "Trying to write beyond buffer length"), D(t3, 34028234663852886e22, -34028234663852886e22)), e4.length <= n3 || i2.write(e4, t3, n3, r3, 23, 4);
            }
            function x(e4, t3, n3, r3, o3) {
              o3 || (d(null != t3, "missing value"), d("boolean" == typeof r3, "missing or invalid endian"), d(null != n3, "missing offset"), d(n3 + 7 < e4.length, "Trying to write beyond buffer length"), D(t3, 17976931348623157e292, -17976931348623157e292)), e4.length <= n3 || i2.write(e4, t3, n3, r3, 52, 8);
            }
            H.Buffer = f2, H.SlowBuffer = f2, H.INSPECT_MAX_BYTES = 50, f2.poolSize = 8192, f2._useTypedArrays = function() {
              try {
                var e4 = new ArrayBuffer(0), t3 = new Uint8Array(e4);
                return t3.foo = function() {
                  return 42;
                }, 42 === t3.foo() && "function" == typeof t3.subarray;
              } catch (e5) {
                return false;
              }
            }(), f2.isEncoding = function(e4) {
              switch (String(e4).toLowerCase()) {
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
            }, f2.isBuffer = function(e4) {
              return !(null == e4 || !e4._isBuffer);
            }, f2.byteLength = function(e4, t3) {
              var n3;
              switch (e4 += "", t3 || "utf8") {
                case "hex":
                  n3 = e4.length / 2;
                  break;
                case "utf8":
                case "utf-8":
                  n3 = T(e4).length;
                  break;
                case "ascii":
                case "binary":
                case "raw":
                  n3 = e4.length;
                  break;
                case "base64":
                  n3 = M(e4).length;
                  break;
                case "ucs2":
                case "ucs-2":
                case "utf16le":
                case "utf-16le":
                  n3 = 2 * e4.length;
                  break;
                default:
                  throw new Error("Unknown encoding");
              }
              return n3;
            }, f2.concat = function(e4, t3) {
              if (d(C(e4), "Usage: Buffer.concat(list, [totalLength])\nlist should be an Array."), 0 === e4.length)
                return new f2(0);
              if (1 === e4.length)
                return e4[0];
              if ("number" != typeof t3)
                for (o3 = t3 = 0; o3 < e4.length; o3++)
                  t3 += e4[o3].length;
              for (var n3 = new f2(t3), r3 = 0, o3 = 0; o3 < e4.length; o3++) {
                var i3 = e4[o3];
                i3.copy(n3, r3), r3 += i3.length;
              }
              return n3;
            }, f2.prototype.write = function(e4, t3, n3, r3) {
              isFinite(t3) ? isFinite(n3) || (r3 = n3, n3 = void 0) : (a3 = r3, r3 = t3, t3 = n3, n3 = a3), t3 = Number(t3) || 0;
              var o3, i3, u3, s3, a3 = this.length - t3;
              switch ((!n3 || a3 < (n3 = Number(n3))) && (n3 = a3), r3 = String(r3 || "utf8").toLowerCase()) {
                case "hex":
                  o3 = function(e5, t4, n4, r4) {
                    n4 = Number(n4) || 0;
                    var o4 = e5.length - n4;
                    (!r4 || o4 < (r4 = Number(r4))) && (r4 = o4), d((o4 = t4.length) % 2 == 0, "Invalid hex string"), o4 / 2 < r4 && (r4 = o4 / 2);
                    for (var i4 = 0; i4 < r4; i4++) {
                      var u4 = parseInt(t4.substr(2 * i4, 2), 16);
                      d(!isNaN(u4), "Invalid hex string"), e5[n4 + i4] = u4;
                    }
                    return f2._charsWritten = 2 * i4, i4;
                  }(this, e4, t3, n3);
                  break;
                case "utf8":
                case "utf-8":
                  i3 = this, u3 = t3, s3 = n3, o3 = f2._charsWritten = c2(T(e4), i3, u3, s3);
                  break;
                case "ascii":
                case "binary":
                  o3 = b(this, e4, t3, n3);
                  break;
                case "base64":
                  i3 = this, u3 = t3, s3 = n3, o3 = f2._charsWritten = c2(M(e4), i3, u3, s3);
                  break;
                case "ucs2":
                case "ucs-2":
                case "utf16le":
                case "utf-16le":
                  o3 = m(this, e4, t3, n3);
                  break;
                default:
                  throw new Error("Unknown encoding");
              }
              return o3;
            }, f2.prototype.toString = function(e4, t3, n3) {
              var r3, o3, i3, u3, s3 = this;
              if (e4 = String(e4 || "utf8").toLowerCase(), t3 = Number(t3) || 0, (n3 = void 0 !== n3 ? Number(n3) : s3.length) === t3)
                return "";
              switch (e4) {
                case "hex":
                  r3 = function(e5, t4, n4) {
                    var r4 = e5.length;
                    (!t4 || t4 < 0) && (t4 = 0);
                    (!n4 || n4 < 0 || r4 < n4) && (n4 = r4);
                    for (var o4 = "", i4 = t4; i4 < n4; i4++)
                      o4 += k(e5[i4]);
                    return o4;
                  }(s3, t3, n3);
                  break;
                case "utf8":
                case "utf-8":
                  r3 = function(e5, t4, n4) {
                    var r4 = "", o4 = "";
                    n4 = Math.min(e5.length, n4);
                    for (var i4 = t4; i4 < n4; i4++)
                      e5[i4] <= 127 ? (r4 += N(o4) + String.fromCharCode(e5[i4]), o4 = "") : o4 += "%" + e5[i4].toString(16);
                    return r4 + N(o4);
                  }(s3, t3, n3);
                  break;
                case "ascii":
                case "binary":
                  r3 = v2(s3, t3, n3);
                  break;
                case "base64":
                  o3 = s3, u3 = n3, r3 = 0 === (i3 = t3) && u3 === o3.length ? a2.fromByteArray(o3) : a2.fromByteArray(o3.slice(i3, u3));
                  break;
                case "ucs2":
                case "ucs-2":
                case "utf16le":
                case "utf-16le":
                  r3 = function(e5, t4, n4) {
                    for (var r4 = e5.slice(t4, n4), o4 = "", i4 = 0; i4 < r4.length; i4 += 2)
                      o4 += String.fromCharCode(r4[i4] + 256 * r4[i4 + 1]);
                    return o4;
                  }(s3, t3, n3);
                  break;
                default:
                  throw new Error("Unknown encoding");
              }
              return r3;
            }, f2.prototype.toJSON = function() {
              return { type: "Buffer", data: Array.prototype.slice.call(this._arr || this, 0) };
            }, f2.prototype.copy = function(e4, t3, n3, r3) {
              if (t3 = t3 || 0, (r3 = r3 || 0 === r3 ? r3 : this.length) !== (n3 = n3 || 0) && 0 !== e4.length && 0 !== this.length) {
                d(n3 <= r3, "sourceEnd < sourceStart"), d(0 <= t3 && t3 < e4.length, "targetStart out of bounds"), d(0 <= n3 && n3 < this.length, "sourceStart out of bounds"), d(0 <= r3 && r3 <= this.length, "sourceEnd out of bounds"), r3 > this.length && (r3 = this.length);
                var o3 = (r3 = e4.length - t3 < r3 - n3 ? e4.length - t3 + n3 : r3) - n3;
                if (o3 < 100 || !f2._useTypedArrays)
                  for (var i3 = 0; i3 < o3; i3++)
                    e4[i3 + t3] = this[i3 + n3];
                else
                  e4._set(this.subarray(n3, n3 + o3), t3);
              }
            }, f2.prototype.slice = function(e4, t3) {
              var n3 = this.length;
              if (e4 = S(e4, n3, 0), t3 = S(t3, n3, n3), f2._useTypedArrays)
                return f2._augment(this.subarray(e4, t3));
              for (var r3 = t3 - e4, o3 = new f2(r3, void 0, true), i3 = 0; i3 < r3; i3++)
                o3[i3] = this[i3 + e4];
              return o3;
            }, f2.prototype.get = function(e4) {
              return console.log(".get() is deprecated. Access using array indexes instead."), this.readUInt8(e4);
            }, f2.prototype.set = function(e4, t3) {
              return console.log(".set() is deprecated. Access using array indexes instead."), this.writeUInt8(e4, t3);
            }, f2.prototype.readUInt8 = function(e4, t3) {
              if (t3 || (d(null != e4, "missing offset"), d(e4 < this.length, "Trying to read beyond buffer length")), !(e4 >= this.length))
                return this[e4];
            }, f2.prototype.readUInt16LE = function(e4, t3) {
              return o2(this, e4, true, t3);
            }, f2.prototype.readUInt16BE = function(e4, t3) {
              return o2(this, e4, false, t3);
            }, f2.prototype.readUInt32LE = function(e4, t3) {
              return u2(this, e4, true, t3);
            }, f2.prototype.readUInt32BE = function(e4, t3) {
              return u2(this, e4, false, t3);
            }, f2.prototype.readInt8 = function(e4, t3) {
              if (t3 || (d(null != e4, "missing offset"), d(e4 < this.length, "Trying to read beyond buffer length")), !(e4 >= this.length))
                return 128 & this[e4] ? -1 * (255 - this[e4] + 1) : this[e4];
            }, f2.prototype.readInt16LE = function(e4, t3) {
              return _(this, e4, true, t3);
            }, f2.prototype.readInt16BE = function(e4, t3) {
              return _(this, e4, false, t3);
            }, f2.prototype.readInt32LE = function(e4, t3) {
              return E(this, e4, true, t3);
            }, f2.prototype.readInt32BE = function(e4, t3) {
              return E(this, e4, false, t3);
            }, f2.prototype.readFloatLE = function(e4, t3) {
              return I(this, e4, true, t3);
            }, f2.prototype.readFloatBE = function(e4, t3) {
              return I(this, e4, false, t3);
            }, f2.prototype.readDoubleLE = function(e4, t3) {
              return A(this, e4, true, t3);
            }, f2.prototype.readDoubleBE = function(e4, t3) {
              return A(this, e4, false, t3);
            }, f2.prototype.writeUInt8 = function(e4, t3, n3) {
              n3 || (d(null != e4, "missing value"), d(null != t3, "missing offset"), d(t3 < this.length, "trying to write beyond buffer length"), Y(e4, 255)), t3 >= this.length || (this[t3] = e4);
            }, f2.prototype.writeUInt16LE = function(e4, t3, n3) {
              s2(this, e4, t3, true, n3);
            }, f2.prototype.writeUInt16BE = function(e4, t3, n3) {
              s2(this, e4, t3, false, n3);
            }, f2.prototype.writeUInt32LE = function(e4, t3, n3) {
              l2(this, e4, t3, true, n3);
            }, f2.prototype.writeUInt32BE = function(e4, t3, n3) {
              l2(this, e4, t3, false, n3);
            }, f2.prototype.writeInt8 = function(e4, t3, n3) {
              n3 || (d(null != e4, "missing value"), d(null != t3, "missing offset"), d(t3 < this.length, "Trying to write beyond buffer length"), F(e4, 127, -128)), t3 >= this.length || (0 <= e4 ? this.writeUInt8(e4, t3, n3) : this.writeUInt8(255 + e4 + 1, t3, n3));
            }, f2.prototype.writeInt16LE = function(e4, t3, n3) {
              B(this, e4, t3, true, n3);
            }, f2.prototype.writeInt16BE = function(e4, t3, n3) {
              B(this, e4, t3, false, n3);
            }, f2.prototype.writeInt32LE = function(e4, t3, n3) {
              L(this, e4, t3, true, n3);
            }, f2.prototype.writeInt32BE = function(e4, t3, n3) {
              L(this, e4, t3, false, n3);
            }, f2.prototype.writeFloatLE = function(e4, t3, n3) {
              U(this, e4, t3, true, n3);
            }, f2.prototype.writeFloatBE = function(e4, t3, n3) {
              U(this, e4, t3, false, n3);
            }, f2.prototype.writeDoubleLE = function(e4, t3, n3) {
              x(this, e4, t3, true, n3);
            }, f2.prototype.writeDoubleBE = function(e4, t3, n3) {
              x(this, e4, t3, false, n3);
            }, f2.prototype.fill = function(e4, t3, n3) {
              if (t3 = t3 || 0, n3 = n3 || this.length, d("number" == typeof (e4 = "string" == typeof (e4 = e4 || 0) ? e4.charCodeAt(0) : e4) && !isNaN(e4), "value is not a number"), d(t3 <= n3, "end < start"), n3 !== t3 && 0 !== this.length) {
                d(0 <= t3 && t3 < this.length, "start out of bounds"), d(0 <= n3 && n3 <= this.length, "end out of bounds");
                for (var r3 = t3; r3 < n3; r3++)
                  this[r3] = e4;
              }
            }, f2.prototype.inspect = function() {
              for (var e4 = [], t3 = this.length, n3 = 0; n3 < t3; n3++)
                if (e4[n3] = k(this[n3]), n3 === H.INSPECT_MAX_BYTES) {
                  e4[n3 + 1] = "...";
                  break;
                }
              return "<Buffer " + e4.join(" ") + ">";
            }, f2.prototype.toArrayBuffer = function() {
              if ("undefined" == typeof Uint8Array)
                throw new Error("Buffer.toArrayBuffer not supported in this browser");
              if (f2._useTypedArrays)
                return new f2(this).buffer;
              for (var e4 = new Uint8Array(this.length), t3 = 0, n3 = e4.length; t3 < n3; t3 += 1)
                e4[t3] = this[t3];
              return e4.buffer;
            };
            var t2 = f2.prototype;
            function S(e4, t3, n3) {
              return "number" != typeof e4 ? n3 : t3 <= (e4 = ~~e4) ? t3 : 0 <= e4 || 0 <= (e4 += t3) ? e4 : 0;
            }
            function j(e4) {
              return (e4 = ~~Math.ceil(+e4)) < 0 ? 0 : e4;
            }
            function C(e4) {
              return (Array.isArray || function(e5) {
                return "[object Array]" === Object.prototype.toString.call(e5);
              })(e4);
            }
            function k(e4) {
              return e4 < 16 ? "0" + e4.toString(16) : e4.toString(16);
            }
            function T(e4) {
              for (var t3 = [], n3 = 0; n3 < e4.length; n3++) {
                var r3 = e4.charCodeAt(n3);
                if (r3 <= 127)
                  t3.push(e4.charCodeAt(n3));
                else
                  for (var o3 = n3, i3 = (55296 <= r3 && r3 <= 57343 && n3++, encodeURIComponent(e4.slice(o3, n3 + 1)).substr(1).split("%")), u3 = 0; u3 < i3.length; u3++)
                    t3.push(parseInt(i3[u3], 16));
              }
              return t3;
            }
            function M(e4) {
              return a2.toByteArray(e4);
            }
            function c2(e4, t3, n3, r3) {
              for (var o3 = 0; o3 < r3 && !(o3 + n3 >= t3.length || o3 >= e4.length); o3++)
                t3[o3 + n3] = e4[o3];
              return o3;
            }
            function N(e4) {
              try {
                return decodeURIComponent(e4);
              } catch (e5) {
                return String.fromCharCode(65533);
              }
            }
            function Y(e4, t3) {
              d("number" == typeof e4, "cannot write a non-number as a number"), d(0 <= e4, "specified a negative value for writing an unsigned value"), d(e4 <= t3, "value is larger than maximum value for type"), d(Math.floor(e4) === e4, "value has a fractional component");
            }
            function F(e4, t3, n3) {
              d("number" == typeof e4, "cannot write a non-number as a number"), d(e4 <= t3, "value larger than maximum allowed value"), d(n3 <= e4, "value smaller than minimum allowed value"), d(Math.floor(e4) === e4, "value has a fractional component");
            }
            function D(e4, t3, n3) {
              d("number" == typeof e4, "cannot write a non-number as a number"), d(e4 <= t3, "value larger than maximum allowed value"), d(n3 <= e4, "value smaller than minimum allowed value");
            }
            function d(e4, t3) {
              if (!e4)
                throw new Error(t3 || "Failed assertion");
            }
            f2._augment = function(e4) {
              return e4._isBuffer = true, e4._get = e4.get, e4._set = e4.set, e4.get = t2.get, e4.set = t2.set, e4.write = t2.write, e4.toString = t2.toString, e4.toLocaleString = t2.toString, e4.toJSON = t2.toJSON, e4.copy = t2.copy, e4.slice = t2.slice, e4.readUInt8 = t2.readUInt8, e4.readUInt16LE = t2.readUInt16LE, e4.readUInt16BE = t2.readUInt16BE, e4.readUInt32LE = t2.readUInt32LE, e4.readUInt32BE = t2.readUInt32BE, e4.readInt8 = t2.readInt8, e4.readInt16LE = t2.readInt16LE, e4.readInt16BE = t2.readInt16BE, e4.readInt32LE = t2.readInt32LE, e4.readInt32BE = t2.readInt32BE, e4.readFloatLE = t2.readFloatLE, e4.readFloatBE = t2.readFloatBE, e4.readDoubleLE = t2.readDoubleLE, e4.readDoubleBE = t2.readDoubleBE, e4.writeUInt8 = t2.writeUInt8, e4.writeUInt16LE = t2.writeUInt16LE, e4.writeUInt16BE = t2.writeUInt16BE, e4.writeUInt32LE = t2.writeUInt32LE, e4.writeUInt32BE = t2.writeUInt32BE, e4.writeInt8 = t2.writeInt8, e4.writeInt16LE = t2.writeInt16LE, e4.writeInt16BE = t2.writeInt16BE, e4.writeInt32LE = t2.writeInt32LE, e4.writeInt32BE = t2.writeInt32BE, e4.writeFloatLE = t2.writeFloatLE, e4.writeFloatBE = t2.writeFloatBE, e4.writeDoubleLE = t2.writeDoubleLE, e4.writeDoubleBE = t2.writeDoubleBE, e4.fill = t2.fill, e4.inspect = t2.inspect, e4.toArrayBuffer = t2.toArrayBuffer, e4;
            };
          }.call(this, O("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, O("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/buffer/index.js", "/node_modules/gulp-browserify/node_modules/buffer");
        }, { "base64-js": 2, buffer: 3, ieee754: 10, lYpoI2: 11 }], 4: [function(c2, d, e2) {
          !function(e3, t2, a2, n2, r2, o2, i2, u2, s2) {
            var a2 = c2("buffer").Buffer, f2 = 4, l2 = new a2(f2);
            l2.fill(0);
            d.exports = { hash: function(e4, t3, n3, r3) {
              for (var o3 = t3(function(e5, t4) {
                e5.length % f2 != 0 && (n4 = e5.length + (f2 - e5.length % f2), e5 = a2.concat([e5, l2], n4));
                for (var n4, r4 = [], o4 = t4 ? e5.readInt32BE : e5.readInt32LE, i4 = 0; i4 < e5.length; i4 += f2)
                  r4.push(o4.call(e5, i4));
                return r4;
              }(e4 = a2.isBuffer(e4) ? e4 : new a2(e4), r3), 8 * e4.length), t3 = r3, i3 = new a2(n3), u3 = t3 ? i3.writeInt32BE : i3.writeInt32LE, s3 = 0; s3 < o3.length; s3++)
                u3.call(i3, o3[s3], 4 * s3, true);
              return i3;
            } };
          }.call(this, c2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/helpers.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { buffer: 3, lYpoI2: 11 }], 5: [function(v2, e2, _) {
          !function(l2, c2, u2, d, h2, p2, g, y2, w) {
            var u2 = v2("buffer").Buffer, e3 = v2("./sha"), t2 = v2("./sha256"), n2 = v2("./rng"), b = { sha1: e3, sha256: t2, md5: v2("./md5") }, s2 = 64, a2 = new u2(s2);
            function r2(e4, n3) {
              var r3 = b[e4 = e4 || "sha1"], o3 = [];
              return r3 || i2("algorithm:", e4, "is not yet supported"), { update: function(e5) {
                return u2.isBuffer(e5) || (e5 = new u2(e5)), o3.push(e5), e5.length, this;
              }, digest: function(e5) {
                var t3 = u2.concat(o3), t3 = n3 ? function(e6, t4, n4) {
                  u2.isBuffer(t4) || (t4 = new u2(t4)), u2.isBuffer(n4) || (n4 = new u2(n4)), t4.length > s2 ? t4 = e6(t4) : t4.length < s2 && (t4 = u2.concat([t4, a2], s2));
                  for (var r4 = new u2(s2), o4 = new u2(s2), i3 = 0; i3 < s2; i3++)
                    r4[i3] = 54 ^ t4[i3], o4[i3] = 92 ^ t4[i3];
                  return n4 = e6(u2.concat([r4, n4])), e6(u2.concat([o4, n4]));
                }(r3, n3, t3) : r3(t3);
                return o3 = null, e5 ? t3.toString(e5) : t3;
              } };
            }
            function i2() {
              var e4 = [].slice.call(arguments).join(" ");
              throw new Error([e4, "we accept pull requests", "http://github.com/dominictarr/crypto-browserify"].join("\n"));
            }
            a2.fill(0), _.createHash = function(e4) {
              return r2(e4);
            }, _.createHmac = r2, _.randomBytes = function(e4, t3) {
              if (!t3 || !t3.call)
                return new u2(n2(e4));
              try {
                t3.call(this, void 0, new u2(n2(e4)));
              } catch (e5) {
                t3(e5);
              }
            };
            var o2, f2 = ["createCredentials", "createCipher", "createCipheriv", "createDecipher", "createDecipheriv", "createSign", "createVerify", "createDiffieHellman", "pbkdf2"], m = function(e4) {
              _[e4] = function() {
                i2("sorry,", e4, "is not implemented yet");
              };
            };
            for (o2 in f2)
              m(f2[o2], o2);
          }.call(this, v2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, v2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/index.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { "./md5": 6, "./rng": 7, "./sha": 8, "./sha256": 9, buffer: 3, lYpoI2: 11 }], 6: [function(w, b, e2) {
          !function(e3, r2, o2, i2, u2, a2, f2, l2, y2) {
            var t2 = w("./helpers");
            function n2(e4, t3) {
              e4[t3 >> 5] |= 128 << t3 % 32, e4[14 + (t3 + 64 >>> 9 << 4)] = t3;
              for (var n3 = 1732584193, r3 = -271733879, o3 = -1732584194, i3 = 271733878, u3 = 0; u3 < e4.length; u3 += 16) {
                var s3 = n3, a3 = r3, f3 = o3, l3 = i3, n3 = c2(n3, r3, o3, i3, e4[u3 + 0], 7, -680876936), i3 = c2(i3, n3, r3, o3, e4[u3 + 1], 12, -389564586), o3 = c2(o3, i3, n3, r3, e4[u3 + 2], 17, 606105819), r3 = c2(r3, o3, i3, n3, e4[u3 + 3], 22, -1044525330);
                n3 = c2(n3, r3, o3, i3, e4[u3 + 4], 7, -176418897), i3 = c2(i3, n3, r3, o3, e4[u3 + 5], 12, 1200080426), o3 = c2(o3, i3, n3, r3, e4[u3 + 6], 17, -1473231341), r3 = c2(r3, o3, i3, n3, e4[u3 + 7], 22, -45705983), n3 = c2(n3, r3, o3, i3, e4[u3 + 8], 7, 1770035416), i3 = c2(i3, n3, r3, o3, e4[u3 + 9], 12, -1958414417), o3 = c2(o3, i3, n3, r3, e4[u3 + 10], 17, -42063), r3 = c2(r3, o3, i3, n3, e4[u3 + 11], 22, -1990404162), n3 = c2(n3, r3, o3, i3, e4[u3 + 12], 7, 1804603682), i3 = c2(i3, n3, r3, o3, e4[u3 + 13], 12, -40341101), o3 = c2(o3, i3, n3, r3, e4[u3 + 14], 17, -1502002290), n3 = d(n3, r3 = c2(r3, o3, i3, n3, e4[u3 + 15], 22, 1236535329), o3, i3, e4[u3 + 1], 5, -165796510), i3 = d(i3, n3, r3, o3, e4[u3 + 6], 9, -1069501632), o3 = d(o3, i3, n3, r3, e4[u3 + 11], 14, 643717713), r3 = d(r3, o3, i3, n3, e4[u3 + 0], 20, -373897302), n3 = d(n3, r3, o3, i3, e4[u3 + 5], 5, -701558691), i3 = d(i3, n3, r3, o3, e4[u3 + 10], 9, 38016083), o3 = d(o3, i3, n3, r3, e4[u3 + 15], 14, -660478335), r3 = d(r3, o3, i3, n3, e4[u3 + 4], 20, -405537848), n3 = d(n3, r3, o3, i3, e4[u3 + 9], 5, 568446438), i3 = d(i3, n3, r3, o3, e4[u3 + 14], 9, -1019803690), o3 = d(o3, i3, n3, r3, e4[u3 + 3], 14, -187363961), r3 = d(r3, o3, i3, n3, e4[u3 + 8], 20, 1163531501), n3 = d(n3, r3, o3, i3, e4[u3 + 13], 5, -1444681467), i3 = d(i3, n3, r3, o3, e4[u3 + 2], 9, -51403784), o3 = d(o3, i3, n3, r3, e4[u3 + 7], 14, 1735328473), n3 = h2(n3, r3 = d(r3, o3, i3, n3, e4[u3 + 12], 20, -1926607734), o3, i3, e4[u3 + 5], 4, -378558), i3 = h2(i3, n3, r3, o3, e4[u3 + 8], 11, -2022574463), o3 = h2(o3, i3, n3, r3, e4[u3 + 11], 16, 1839030562), r3 = h2(r3, o3, i3, n3, e4[u3 + 14], 23, -35309556), n3 = h2(n3, r3, o3, i3, e4[u3 + 1], 4, -1530992060), i3 = h2(i3, n3, r3, o3, e4[u3 + 4], 11, 1272893353), o3 = h2(o3, i3, n3, r3, e4[u3 + 7], 16, -155497632), r3 = h2(r3, o3, i3, n3, e4[u3 + 10], 23, -1094730640), n3 = h2(n3, r3, o3, i3, e4[u3 + 13], 4, 681279174), i3 = h2(i3, n3, r3, o3, e4[u3 + 0], 11, -358537222), o3 = h2(o3, i3, n3, r3, e4[u3 + 3], 16, -722521979), r3 = h2(r3, o3, i3, n3, e4[u3 + 6], 23, 76029189), n3 = h2(n3, r3, o3, i3, e4[u3 + 9], 4, -640364487), i3 = h2(i3, n3, r3, o3, e4[u3 + 12], 11, -421815835), o3 = h2(o3, i3, n3, r3, e4[u3 + 15], 16, 530742520), n3 = p2(n3, r3 = h2(r3, o3, i3, n3, e4[u3 + 2], 23, -995338651), o3, i3, e4[u3 + 0], 6, -198630844), i3 = p2(i3, n3, r3, o3, e4[u3 + 7], 10, 1126891415), o3 = p2(o3, i3, n3, r3, e4[u3 + 14], 15, -1416354905), r3 = p2(r3, o3, i3, n3, e4[u3 + 5], 21, -57434055), n3 = p2(n3, r3, o3, i3, e4[u3 + 12], 6, 1700485571), i3 = p2(i3, n3, r3, o3, e4[u3 + 3], 10, -1894986606), o3 = p2(o3, i3, n3, r3, e4[u3 + 10], 15, -1051523), r3 = p2(r3, o3, i3, n3, e4[u3 + 1], 21, -2054922799), n3 = p2(n3, r3, o3, i3, e4[u3 + 8], 6, 1873313359), i3 = p2(i3, n3, r3, o3, e4[u3 + 15], 10, -30611744), o3 = p2(o3, i3, n3, r3, e4[u3 + 6], 15, -1560198380), r3 = p2(r3, o3, i3, n3, e4[u3 + 13], 21, 1309151649), n3 = p2(n3, r3, o3, i3, e4[u3 + 4], 6, -145523070), i3 = p2(i3, n3, r3, o3, e4[u3 + 11], 10, -1120210379), o3 = p2(o3, i3, n3, r3, e4[u3 + 2], 15, 718787259), r3 = p2(r3, o3, i3, n3, e4[u3 + 9], 21, -343485551), n3 = g(n3, s3), r3 = g(r3, a3), o3 = g(o3, f3), i3 = g(i3, l3);
              }
              return Array(n3, r3, o3, i3);
            }
            function s2(e4, t3, n3, r3, o3, i3) {
              return g((t3 = g(g(t3, e4), g(r3, i3))) << o3 | t3 >>> 32 - o3, n3);
            }
            function c2(e4, t3, n3, r3, o3, i3, u3) {
              return s2(t3 & n3 | ~t3 & r3, e4, t3, o3, i3, u3);
            }
            function d(e4, t3, n3, r3, o3, i3, u3) {
              return s2(t3 & r3 | n3 & ~r3, e4, t3, o3, i3, u3);
            }
            function h2(e4, t3, n3, r3, o3, i3, u3) {
              return s2(t3 ^ n3 ^ r3, e4, t3, o3, i3, u3);
            }
            function p2(e4, t3, n3, r3, o3, i3, u3) {
              return s2(n3 ^ (t3 | ~r3), e4, t3, o3, i3, u3);
            }
            function g(e4, t3) {
              var n3 = (65535 & e4) + (65535 & t3);
              return (e4 >> 16) + (t3 >> 16) + (n3 >> 16) << 16 | 65535 & n3;
            }
            b.exports = function(e4) {
              return t2.hash(e4, n2, 16);
            };
          }.call(this, w("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, w("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/md5.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 7: [function(e2, l2, t2) {
          !function(e3, t3, n2, r2, o2, i2, u2, s2, f2) {
            var a2;
            l2.exports = a2 || function(e4) {
              for (var t4, n3 = new Array(e4), r3 = 0; r3 < e4; r3++)
                0 == (3 & r3) && (t4 = 4294967296 * Math.random()), n3[r3] = t4 >>> ((3 & r3) << 3) & 255;
              return n3;
            };
          }.call(this, e2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/rng.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { buffer: 3, lYpoI2: 11 }], 8: [function(c2, d, e2) {
          !function(e3, t2, n2, r2, o2, s2, a2, f2, l2) {
            var i2 = c2("./helpers");
            function u2(l3, c3) {
              l3[c3 >> 5] |= 128 << 24 - c3 % 32, l3[15 + (c3 + 64 >> 9 << 4)] = c3;
              for (var e4, t3, n3, r3 = Array(80), o3 = 1732584193, i3 = -271733879, u3 = -1732584194, s3 = 271733878, d2 = -1009589776, h2 = 0; h2 < l3.length; h2 += 16) {
                for (var p2 = o3, g = i3, y2 = u3, w = s3, b = d2, a3 = 0; a3 < 80; a3++) {
                  r3[a3] = a3 < 16 ? l3[h2 + a3] : v2(r3[a3 - 3] ^ r3[a3 - 8] ^ r3[a3 - 14] ^ r3[a3 - 16], 1);
                  var f3 = m(m(v2(o3, 5), (f3 = i3, t3 = u3, n3 = s3, (e4 = a3) < 20 ? f3 & t3 | ~f3 & n3 : !(e4 < 40) && e4 < 60 ? f3 & t3 | f3 & n3 | t3 & n3 : f3 ^ t3 ^ n3)), m(m(d2, r3[a3]), (e4 = a3) < 20 ? 1518500249 : e4 < 40 ? 1859775393 : e4 < 60 ? -1894007588 : -899497514)), d2 = s3, s3 = u3, u3 = v2(i3, 30), i3 = o3, o3 = f3;
                }
                o3 = m(o3, p2), i3 = m(i3, g), u3 = m(u3, y2), s3 = m(s3, w), d2 = m(d2, b);
              }
              return Array(o3, i3, u3, s3, d2);
            }
            function m(e4, t3) {
              var n3 = (65535 & e4) + (65535 & t3);
              return (e4 >> 16) + (t3 >> 16) + (n3 >> 16) << 16 | 65535 & n3;
            }
            function v2(e4, t3) {
              return e4 << t3 | e4 >>> 32 - t3;
            }
            d.exports = function(e4) {
              return i2.hash(e4, u2, 20, true);
            };
          }.call(this, c2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/sha.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 9: [function(c2, d, e2) {
          !function(e3, t2, n2, r2, u2, s2, a2, f2, l2) {
            function b(e4, t3) {
              var n3 = (65535 & e4) + (65535 & t3);
              return (e4 >> 16) + (t3 >> 16) + (n3 >> 16) << 16 | 65535 & n3;
            }
            function o2(e4, l3) {
              var c3, d2 = new Array(1116352408, 1899447441, 3049323471, 3921009573, 961987163, 1508970993, 2453635748, 2870763221, 3624381080, 310598401, 607225278, 1426881987, 1925078388, 2162078206, 2614888103, 3248222580, 3835390401, 4022224774, 264347078, 604807628, 770255983, 1249150122, 1555081692, 1996064986, 2554220882, 2821834349, 2952996808, 3210313671, 3336571891, 3584528711, 113926993, 338241895, 666307205, 773529912, 1294757372, 1396182291, 1695183700, 1986661051, 2177026350, 2456956037, 2730485921, 2820302411, 3259730800, 3345764771, 3516065817, 3600352804, 4094571909, 275423344, 430227734, 506948616, 659060556, 883997877, 958139571, 1322822218, 1537002063, 1747873779, 1955562222, 2024104815, 2227730452, 2361852424, 2428436474, 2756734187, 3204031479, 3329325298), t3 = new Array(1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225), n3 = new Array(64);
              e4[l3 >> 5] |= 128 << 24 - l3 % 32, e4[15 + (l3 + 64 >> 9 << 4)] = l3;
              for (var r3, o3, h2 = 0; h2 < e4.length; h2 += 16) {
                for (var i3 = t3[0], u3 = t3[1], s3 = t3[2], p2 = t3[3], a3 = t3[4], g = t3[5], y2 = t3[6], w = t3[7], f3 = 0; f3 < 64; f3++)
                  n3[f3] = f3 < 16 ? e4[f3 + h2] : b(b(b((o3 = n3[f3 - 2], m(o3, 17) ^ m(o3, 19) ^ v2(o3, 10)), n3[f3 - 7]), (o3 = n3[f3 - 15], m(o3, 7) ^ m(o3, 18) ^ v2(o3, 3))), n3[f3 - 16]), c3 = b(b(b(b(w, m(o3 = a3, 6) ^ m(o3, 11) ^ m(o3, 25)), a3 & g ^ ~a3 & y2), d2[f3]), n3[f3]), r3 = b(m(r3 = i3, 2) ^ m(r3, 13) ^ m(r3, 22), i3 & u3 ^ i3 & s3 ^ u3 & s3), w = y2, y2 = g, g = a3, a3 = b(p2, c3), p2 = s3, s3 = u3, u3 = i3, i3 = b(c3, r3);
                t3[0] = b(i3, t3[0]), t3[1] = b(u3, t3[1]), t3[2] = b(s3, t3[2]), t3[3] = b(p2, t3[3]), t3[4] = b(a3, t3[4]), t3[5] = b(g, t3[5]), t3[6] = b(y2, t3[6]), t3[7] = b(w, t3[7]);
              }
              return t3;
            }
            var i2 = c2("./helpers"), m = function(e4, t3) {
              return e4 >>> t3 | e4 << 32 - t3;
            }, v2 = function(e4, t3) {
              return e4 >>> t3;
            };
            d.exports = function(e4) {
              return i2.hash(e4, o2, 32, true);
            };
          }.call(this, c2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, c2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/crypto-browserify/sha256.js", "/node_modules/gulp-browserify/node_modules/crypto-browserify");
        }, { "./helpers": 4, buffer: 3, lYpoI2: 11 }], 10: [function(e2, t2, f2) {
          !function(e3, t3, n2, r2, o2, i2, u2, s2, a2) {
            f2.read = function(e4, t4, n3, r3, o3) {
              var i3, u3, l2 = 8 * o3 - r3 - 1, c2 = (1 << l2) - 1, d = c2 >> 1, s3 = -7, a3 = n3 ? o3 - 1 : 0, f3 = n3 ? -1 : 1, o3 = e4[t4 + a3];
              for (a3 += f3, i3 = o3 & (1 << -s3) - 1, o3 >>= -s3, s3 += l2; 0 < s3; i3 = 256 * i3 + e4[t4 + a3], a3 += f3, s3 -= 8)
                ;
              for (u3 = i3 & (1 << -s3) - 1, i3 >>= -s3, s3 += r3; 0 < s3; u3 = 256 * u3 + e4[t4 + a3], a3 += f3, s3 -= 8)
                ;
              if (0 === i3)
                i3 = 1 - d;
              else {
                if (i3 === c2)
                  return u3 ? NaN : 1 / 0 * (o3 ? -1 : 1);
                u3 += Math.pow(2, r3), i3 -= d;
              }
              return (o3 ? -1 : 1) * u3 * Math.pow(2, i3 - r3);
            }, f2.write = function(e4, t4, l2, n3, r3, c2) {
              var o3, i3, u3 = 8 * c2 - r3 - 1, s3 = (1 << u3) - 1, a3 = s3 >> 1, d = 23 === r3 ? Math.pow(2, -24) - Math.pow(2, -77) : 0, f3 = n3 ? 0 : c2 - 1, h2 = n3 ? 1 : -1, c2 = t4 < 0 || 0 === t4 && 1 / t4 < 0 ? 1 : 0;
              for (t4 = Math.abs(t4), isNaN(t4) || t4 === 1 / 0 ? (i3 = isNaN(t4) ? 1 : 0, o3 = s3) : (o3 = Math.floor(Math.log(t4) / Math.LN2), t4 * (n3 = Math.pow(2, -o3)) < 1 && (o3--, n3 *= 2), 2 <= (t4 += 1 <= o3 + a3 ? d / n3 : d * Math.pow(2, 1 - a3)) * n3 && (o3++, n3 /= 2), s3 <= o3 + a3 ? (i3 = 0, o3 = s3) : 1 <= o3 + a3 ? (i3 = (t4 * n3 - 1) * Math.pow(2, r3), o3 += a3) : (i3 = t4 * Math.pow(2, a3 - 1) * Math.pow(2, r3), o3 = 0)); 8 <= r3; e4[l2 + f3] = 255 & i3, f3 += h2, i3 /= 256, r3 -= 8)
                ;
              for (o3 = o3 << r3 | i3, u3 += r3; 0 < u3; e4[l2 + f3] = 255 & o3, f3 += h2, o3 /= 256, u3 -= 8)
                ;
              e4[l2 + f3 - h2] |= 128 * c2;
            };
          }.call(this, e2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/ieee754/index.js", "/node_modules/gulp-browserify/node_modules/ieee754");
        }, { buffer: 3, lYpoI2: 11 }], 11: [function(e2, h2, t2) {
          !function(e3, t3, n2, r2, o2, f2, l2, c2, d) {
            var i2, u2, s2;
            function a2() {
            }
            (e3 = h2.exports = {}).nextTick = (u2 = "undefined" != typeof window && window.setImmediate, s2 = "undefined" != typeof window && window.postMessage && window.addEventListener, u2 ? function(e4) {
              return window.setImmediate(e4);
            } : s2 ? (i2 = [], window.addEventListener("message", function(e4) {
              var t4 = e4.source;
              t4 !== window && null !== t4 || "process-tick" !== e4.data || (e4.stopPropagation(), 0 < i2.length && i2.shift()());
            }, true), function(e4) {
              i2.push(e4), window.postMessage("process-tick", "*");
            }) : function(e4) {
              setTimeout(e4, 0);
            }), e3.title = "browser", e3.browser = true, e3.env = {}, e3.argv = [], e3.on = a2, e3.addListener = a2, e3.once = a2, e3.off = a2, e3.removeListener = a2, e3.removeAllListeners = a2, e3.emit = a2, e3.binding = function(e4) {
              throw new Error("process.binding is not supported");
            }, e3.cwd = function() {
              return "/";
            }, e3.chdir = function(e4) {
              throw new Error("process.chdir is not supported");
            };
          }.call(this, e2("lYpoI2"), "undefined" != typeof self ? self : "undefined" != typeof window ? window : {}, e2("buffer").Buffer, arguments[3], arguments[4], arguments[5], arguments[6], "/node_modules/gulp-browserify/node_modules/process/browser.js", "/node_modules/gulp-browserify/node_modules/process");
        }, { buffer: 3, lYpoI2: 11 }] }, {}, [1])(1);
      });
    }
  });

  // node_modules/object-sizeof/byte_size.js
  var require_byte_size = __commonJS({
    "node_modules/object-sizeof/byte_size.js"(exports2, module) {
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
    "node_modules/base64-js/index.js"(exports2) {
      "use strict";
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      exports2.byteLength = byteLength;
      exports2.toByteArray = toByteArray;
      exports2.fromByteArray = fromByteArray;
      var lookup = [];
      var revLookup = [];
      var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
      var code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      for (i2 = 0, len = code.length; i2 < len; ++i2) {
        lookup[i2] = code[i2];
        revLookup[code.charCodeAt(i2)] = i2;
      }
      var i2;
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
        var i3;
        for (i3 = 0; i3 < len2; i3 += 4) {
          tmp = revLookup[b64.charCodeAt(i3)] << 18 | revLookup[b64.charCodeAt(i3 + 1)] << 12 | revLookup[b64.charCodeAt(i3 + 2)] << 6 | revLookup[b64.charCodeAt(i3 + 3)];
          arr[curByte++] = tmp >> 16 & 255;
          arr[curByte++] = tmp >> 8 & 255;
          arr[curByte++] = tmp & 255;
        }
        if (placeHoldersLen === 2) {
          tmp = revLookup[b64.charCodeAt(i3)] << 2 | revLookup[b64.charCodeAt(i3 + 1)] >> 4;
          arr[curByte++] = tmp & 255;
        }
        if (placeHoldersLen === 1) {
          tmp = revLookup[b64.charCodeAt(i3)] << 10 | revLookup[b64.charCodeAt(i3 + 1)] << 4 | revLookup[b64.charCodeAt(i3 + 2)] >> 2;
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
        for (var i3 = start; i3 < end; i3 += 3) {
          tmp = (uint8[i3] << 16 & 16711680) + (uint8[i3 + 1] << 8 & 65280) + (uint8[i3 + 2] & 255);
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
        for (var i3 = 0, len22 = len2 - extraBytes; i3 < len22; i3 += maxChunkLength) {
          parts.push(encodeChunk(uint8, i3, i3 + maxChunkLength > len22 ? len22 : i3 + maxChunkLength));
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
    "node_modules/ieee754/index.js"(exports2) {
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      exports2.read = function(buffer, offset, isLE, mLen, nBytes) {
        var e2, m;
        var eLen = nBytes * 8 - mLen - 1;
        var eMax = (1 << eLen) - 1;
        var eBias = eMax >> 1;
        var nBits = -7;
        var i2 = isLE ? nBytes - 1 : 0;
        var d = isLE ? -1 : 1;
        var s2 = buffer[offset + i2];
        i2 += d;
        e2 = s2 & (1 << -nBits) - 1;
        s2 >>= -nBits;
        nBits += eLen;
        for (; nBits > 0; e2 = e2 * 256 + buffer[offset + i2], i2 += d, nBits -= 8) {
        }
        m = e2 & (1 << -nBits) - 1;
        e2 >>= -nBits;
        nBits += mLen;
        for (; nBits > 0; m = m * 256 + buffer[offset + i2], i2 += d, nBits -= 8) {
        }
        if (e2 === 0) {
          e2 = 1 - eBias;
        } else if (e2 === eMax) {
          return m ? NaN : (s2 ? -1 : 1) * Infinity;
        } else {
          m = m + Math.pow(2, mLen);
          e2 = e2 - eBias;
        }
        return (s2 ? -1 : 1) * m * Math.pow(2, e2 - mLen);
      };
      exports2.write = function(buffer, value, offset, isLE, mLen, nBytes) {
        var e2, m, c2;
        var eLen = nBytes * 8 - mLen - 1;
        var eMax = (1 << eLen) - 1;
        var eBias = eMax >> 1;
        var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
        var i2 = isLE ? 0 : nBytes - 1;
        var d = isLE ? 1 : -1;
        var s2 = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
        value = Math.abs(value);
        if (isNaN(value) || value === Infinity) {
          m = isNaN(value) ? 1 : 0;
          e2 = eMax;
        } else {
          e2 = Math.floor(Math.log(value) / Math.LN2);
          if (value * (c2 = Math.pow(2, -e2)) < 1) {
            e2--;
            c2 *= 2;
          }
          if (e2 + eBias >= 1) {
            value += rt / c2;
          } else {
            value += rt * Math.pow(2, 1 - eBias);
          }
          if (value * c2 >= 2) {
            e2++;
            c2 /= 2;
          }
          if (e2 + eBias >= eMax) {
            m = 0;
            e2 = eMax;
          } else if (e2 + eBias >= 1) {
            m = (value * c2 - 1) * Math.pow(2, mLen);
            e2 = e2 + eBias;
          } else {
            m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
            e2 = 0;
          }
        }
        for (; mLen >= 8; buffer[offset + i2] = m & 255, i2 += d, m /= 256, mLen -= 8) {
        }
        e2 = e2 << mLen | m;
        eLen += mLen;
        for (; eLen > 0; buffer[offset + i2] = e2 & 255, i2 += d, e2 /= 256, eLen -= 8) {
        }
        buffer[offset + i2 - d] |= s2 * 128;
      };
    }
  });

  // node_modules/buffer/index.js
  var require_buffer = __commonJS({
    "node_modules/buffer/index.js"(exports2) {
      "use strict";
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      var base642 = require_base64_js();
      var ieee754 = require_ieee754();
      var customInspectSymbol = typeof Symbol === "function" && typeof Symbol["for"] === "function" ? Symbol["for"]("nodejs.util.inspect.custom") : null;
      exports2.Buffer = Buffer3;
      exports2.SlowBuffer = SlowBuffer;
      exports2.INSPECT_MAX_BYTES = 50;
      var K_MAX_LENGTH = 2147483647;
      exports2.kMaxLength = K_MAX_LENGTH;
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
        } catch (e2) {
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
        const b = fromObject(value);
        if (b)
          return b;
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
        for (let i2 = 0; i2 < length; i2 += 1) {
          buf[i2] = array[i2] & 255;
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
      Buffer3.isBuffer = function isBuffer2(b) {
        return b != null && b._isBuffer === true && b !== Buffer3.prototype;
      };
      Buffer3.compare = function compare(a2, b) {
        if (isInstance(a2, Uint8Array))
          a2 = Buffer3.from(a2, a2.offset, a2.byteLength);
        if (isInstance(b, Uint8Array))
          b = Buffer3.from(b, b.offset, b.byteLength);
        if (!Buffer3.isBuffer(a2) || !Buffer3.isBuffer(b)) {
          throw new TypeError(
            'The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array'
          );
        }
        if (a2 === b)
          return 0;
        let x = a2.length;
        let y2 = b.length;
        for (let i2 = 0, len = Math.min(x, y2); i2 < len; ++i2) {
          if (a2[i2] !== b[i2]) {
            x = a2[i2];
            y2 = b[i2];
            break;
          }
        }
        if (x < y2)
          return -1;
        if (y2 < x)
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
        let i2;
        if (length === void 0) {
          length = 0;
          for (i2 = 0; i2 < list.length; ++i2) {
            length += list[i2].length;
          }
        }
        const buffer = Buffer3.allocUnsafe(length);
        let pos = 0;
        for (i2 = 0; i2 < list.length; ++i2) {
          let buf = list[i2];
          if (isInstance(buf, Uint8Array)) {
            if (pos + buf.length > buffer.length) {
              if (!Buffer3.isBuffer(buf))
                buf = Buffer3.from(buf);
              buf.copy(buffer, pos);
            } else {
              Uint8Array.prototype.set.call(
                buffer,
                buf,
                pos
              );
            }
          } else if (!Buffer3.isBuffer(buf)) {
            throw new TypeError('"list" argument must be an Array of Buffers');
          } else {
            buf.copy(buffer, pos);
          }
          pos += buf.length;
        }
        return buffer;
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
      function swap(b, n2, m) {
        const i2 = b[n2];
        b[n2] = b[m];
        b[m] = i2;
      }
      Buffer3.prototype.swap16 = function swap16() {
        const len = this.length;
        if (len % 2 !== 0) {
          throw new RangeError("Buffer size must be a multiple of 16-bits");
        }
        for (let i2 = 0; i2 < len; i2 += 2) {
          swap(this, i2, i2 + 1);
        }
        return this;
      };
      Buffer3.prototype.swap32 = function swap32() {
        const len = this.length;
        if (len % 4 !== 0) {
          throw new RangeError("Buffer size must be a multiple of 32-bits");
        }
        for (let i2 = 0; i2 < len; i2 += 4) {
          swap(this, i2, i2 + 3);
          swap(this, i2 + 1, i2 + 2);
        }
        return this;
      };
      Buffer3.prototype.swap64 = function swap64() {
        const len = this.length;
        if (len % 8 !== 0) {
          throw new RangeError("Buffer size must be a multiple of 64-bits");
        }
        for (let i2 = 0; i2 < len; i2 += 8) {
          swap(this, i2, i2 + 7);
          swap(this, i2 + 1, i2 + 6);
          swap(this, i2 + 2, i2 + 5);
          swap(this, i2 + 3, i2 + 4);
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
      Buffer3.prototype.equals = function equals(b) {
        if (!Buffer3.isBuffer(b))
          throw new TypeError("Argument must be a Buffer");
        if (this === b)
          return true;
        return Buffer3.compare(this, b) === 0;
      };
      Buffer3.prototype.inspect = function inspect() {
        let str = "";
        const max = exports2.INSPECT_MAX_BYTES;
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
        let x = thisEnd - thisStart;
        let y2 = end - start;
        const len = Math.min(x, y2);
        const thisCopy = this.slice(thisStart, thisEnd);
        const targetCopy = target.slice(start, end);
        for (let i2 = 0; i2 < len; ++i2) {
          if (thisCopy[i2] !== targetCopy[i2]) {
            x = thisCopy[i2];
            y2 = targetCopy[i2];
            break;
          }
        }
        if (x < y2)
          return -1;
        if (y2 < x)
          return 1;
        return 0;
      };
      function bidirectionalIndexOf(buffer, val, byteOffset, encoding, dir) {
        if (buffer.length === 0)
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
          byteOffset = dir ? 0 : buffer.length - 1;
        }
        if (byteOffset < 0)
          byteOffset = buffer.length + byteOffset;
        if (byteOffset >= buffer.length) {
          if (dir)
            return -1;
          else
            byteOffset = buffer.length - 1;
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
          return arrayIndexOf(buffer, val, byteOffset, encoding, dir);
        } else if (typeof val === "number") {
          val = val & 255;
          if (typeof Uint8Array.prototype.indexOf === "function") {
            if (dir) {
              return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset);
            } else {
              return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset);
            }
          }
          return arrayIndexOf(buffer, [val], byteOffset, encoding, dir);
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
        function read(buf, i3) {
          if (indexSize === 1) {
            return buf[i3];
          } else {
            return buf.readUInt16BE(i3 * indexSize);
          }
        }
        let i2;
        if (dir) {
          let foundIndex = -1;
          for (i2 = byteOffset; i2 < arrLength; i2++) {
            if (read(arr, i2) === read(val, foundIndex === -1 ? 0 : i2 - foundIndex)) {
              if (foundIndex === -1)
                foundIndex = i2;
              if (i2 - foundIndex + 1 === valLength)
                return foundIndex * indexSize;
            } else {
              if (foundIndex !== -1)
                i2 -= i2 - foundIndex;
              foundIndex = -1;
            }
          }
        } else {
          if (byteOffset + valLength > arrLength)
            byteOffset = arrLength - valLength;
          for (i2 = byteOffset; i2 >= 0; i2--) {
            let found = true;
            for (let j = 0; j < valLength; j++) {
              if (read(arr, i2 + j) !== read(val, j)) {
                found = false;
                break;
              }
            }
            if (found)
              return i2;
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
        let i2;
        for (i2 = 0; i2 < length; ++i2) {
          const parsed = parseInt(string.substr(i2 * 2, 2), 16);
          if (numberIsNaN(parsed))
            return i2;
          buf[offset + i2] = parsed;
        }
        return i2;
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
      Buffer3.prototype.write = function write(string, offset, length, encoding) {
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
        let i2 = start;
        while (i2 < end) {
          const firstByte = buf[i2];
          let codePoint = null;
          let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
          if (i2 + bytesPerSequence <= end) {
            let secondByte, thirdByte, fourthByte, tempCodePoint;
            switch (bytesPerSequence) {
              case 1:
                if (firstByte < 128) {
                  codePoint = firstByte;
                }
                break;
              case 2:
                secondByte = buf[i2 + 1];
                if ((secondByte & 192) === 128) {
                  tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
                  if (tempCodePoint > 127) {
                    codePoint = tempCodePoint;
                  }
                }
                break;
              case 3:
                secondByte = buf[i2 + 1];
                thirdByte = buf[i2 + 2];
                if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
                  tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
                  if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                    codePoint = tempCodePoint;
                  }
                }
                break;
              case 4:
                secondByte = buf[i2 + 1];
                thirdByte = buf[i2 + 2];
                fourthByte = buf[i2 + 3];
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
          i2 += bytesPerSequence;
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
        let i2 = 0;
        while (i2 < len) {
          res += String.fromCharCode.apply(
            String,
            codePoints.slice(i2, i2 += MAX_ARGUMENTS_LENGTH)
          );
        }
        return res;
      }
      function asciiSlice(buf, start, end) {
        let ret = "";
        end = Math.min(buf.length, end);
        for (let i2 = start; i2 < end; ++i2) {
          ret += String.fromCharCode(buf[i2] & 127);
        }
        return ret;
      }
      function latin1Slice(buf, start, end) {
        let ret = "";
        end = Math.min(buf.length, end);
        for (let i2 = start; i2 < end; ++i2) {
          ret += String.fromCharCode(buf[i2]);
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
        for (let i2 = start; i2 < end; ++i2) {
          out += hexSliceLookupTable[buf[i2]];
        }
        return out;
      }
      function utf16leSlice(buf, start, end) {
        const bytes = buf.slice(start, end);
        let res = "";
        for (let i2 = 0; i2 < bytes.length - 1; i2 += 2) {
          res += String.fromCharCode(bytes[i2] + bytes[i2 + 1] * 256);
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
        let i2 = 0;
        while (++i2 < byteLength2 && (mul *= 256)) {
          val += this[offset + i2] * mul;
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
        let i2 = 0;
        while (++i2 < byteLength2 && (mul *= 256)) {
          val += this[offset + i2] * mul;
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
        let i2 = byteLength2;
        let mul = 1;
        let val = this[offset + --i2];
        while (i2 > 0 && (mul *= 256)) {
          val += this[offset + --i2] * mul;
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
        let i2 = 0;
        this[offset] = value & 255;
        while (++i2 < byteLength2 && (mul *= 256)) {
          this[offset + i2] = value / mul & 255;
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
        let i2 = byteLength2 - 1;
        let mul = 1;
        this[offset + i2] = value & 255;
        while (--i2 >= 0 && (mul *= 256)) {
          this[offset + i2] = value / mul & 255;
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
        let i2 = 0;
        let mul = 1;
        let sub = 0;
        this[offset] = value & 255;
        while (++i2 < byteLength2 && (mul *= 256)) {
          if (value < 0 && sub === 0 && this[offset + i2 - 1] !== 0) {
            sub = 1;
          }
          this[offset + i2] = (value / mul >> 0) - sub & 255;
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
        let i2 = byteLength2 - 1;
        let mul = 1;
        let sub = 0;
        this[offset + i2] = value & 255;
        while (--i2 >= 0 && (mul *= 256)) {
          if (value < 0 && sub === 0 && this[offset + i2 + 1] !== 0) {
            sub = 1;
          }
          this[offset + i2] = (value / mul >> 0) - sub & 255;
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
          checkIEEE754(buf, value, offset, 4, 34028234663852886e22, -34028234663852886e22);
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
          checkIEEE754(buf, value, offset, 8, 17976931348623157e292, -17976931348623157e292);
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
        let i2;
        if (typeof val === "number") {
          for (i2 = start; i2 < end; ++i2) {
            this[i2] = val;
          }
        } else {
          const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
          const len = bytes.length;
          if (len === 0) {
            throw new TypeError('The value "' + val + '" is invalid for argument "value"');
          }
          for (i2 = 0; i2 < end - start; ++i2) {
            this[i2 + start] = bytes[i2 % len];
          }
        }
        return this;
      };
      var errors = {};
      function E(sym, getMessage, Base) {
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
      E(
        "ERR_BUFFER_OUT_OF_BOUNDS",
        function(name2) {
          if (name2) {
            return `${name2} is outside of buffer bounds`;
          }
          return "Attempt to access memory outside buffer bounds";
        },
        RangeError
      );
      E(
        "ERR_INVALID_ARG_TYPE",
        function(name2, actual) {
          return `The "${name2}" argument must be of type number. Received type ${typeof actual}`;
        },
        TypeError
      );
      E(
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
        let i2 = val.length;
        const start = val[0] === "-" ? 1 : 0;
        for (; i2 >= start + 4; i2 -= 3) {
          res = `_${val.slice(i2 - 3, i2)}${res}`;
        }
        return `${val.slice(0, i2)}${res}`;
      }
      function checkBounds(buf, offset, byteLength2) {
        validateNumber(offset, "offset");
        if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
          boundsError(offset, buf.length - (byteLength2 + 1));
        }
      }
      function checkIntBI(value, min, max, buf, offset, byteLength2) {
        if (value > max || value < min) {
          const n2 = typeof min === "bigint" ? "n" : "";
          let range;
          if (byteLength2 > 3) {
            if (min === 0 || min === BigInt(0)) {
              range = `>= 0${n2} and < 2${n2} ** ${(byteLength2 + 1) * 8}${n2}`;
            } else {
              range = `>= -(2${n2} ** ${(byteLength2 + 1) * 8 - 1}${n2}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n2}`;
            }
          } else {
            range = `>= ${min}${n2} and <= ${max}${n2}`;
          }
          throw new errors.ERR_OUT_OF_RANGE("value", range, value);
        }
        checkBounds(buf, offset, byteLength2);
      }
      function validateNumber(value, name2) {
        if (typeof value !== "number") {
          throw new errors.ERR_INVALID_ARG_TYPE(name2, "number", value);
        }
      }
      function boundsError(value, length, type) {
        if (Math.floor(value) !== value) {
          validateNumber(value, type);
          throw new errors.ERR_OUT_OF_RANGE(type || "offset", "an integer", value);
        }
        if (length < 0) {
          throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
        }
        throw new errors.ERR_OUT_OF_RANGE(
          type || "offset",
          `>= ${type ? 1 : 0} and <= ${length}`,
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
        for (let i2 = 0; i2 < length; ++i2) {
          codePoint = string.charCodeAt(i2);
          if (codePoint > 55295 && codePoint < 57344) {
            if (!leadSurrogate) {
              if (codePoint > 56319) {
                if ((units -= 3) > -1)
                  bytes.push(239, 191, 189);
                continue;
              } else if (i2 + 1 === length) {
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
        for (let i2 = 0; i2 < str.length; ++i2) {
          byteArray.push(str.charCodeAt(i2) & 255);
        }
        return byteArray;
      }
      function utf16leToBytes(str, units) {
        let c2, hi, lo;
        const byteArray = [];
        for (let i2 = 0; i2 < str.length; ++i2) {
          if ((units -= 2) < 0)
            break;
          c2 = str.charCodeAt(i2);
          hi = c2 >> 8;
          lo = c2 % 256;
          byteArray.push(lo);
          byteArray.push(hi);
        }
        return byteArray;
      }
      function base64ToBytes(str) {
        return base642.toByteArray(base64clean(str));
      }
      function blitBuffer(src, dst, offset, length) {
        let i2;
        for (i2 = 0; i2 < length; ++i2) {
          if (i2 + offset >= dst.length || i2 >= src.length)
            break;
          dst[i2 + offset] = src[i2];
        }
        return i2;
      }
      function isInstance(obj, type) {
        return obj instanceof type || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type.name;
      }
      function numberIsNaN(obj) {
        return obj !== obj;
      }
      var hexSliceLookupTable = function() {
        const alphabet = "0123456789abcdef";
        const table = new Array(256);
        for (let i2 = 0; i2 < 16; ++i2) {
          const i16 = i2 * 16;
          for (let j = 0; j < 16; ++j) {
            table[i16 + j] = alphabet[i2] + alphabet[j];
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
    "node_modules/object-sizeof/indexv2.js"(exports2, module) {
      "use strict";
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
          const buffer = new Buffer3.from(objectToString);
          totalSize = buffer.byteLength;
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
            for (const i2 in value) {
              stack.push(value[i2]);
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

  // node_modules/form-data/lib/browser.js
  var require_browser = __commonJS({
    "node_modules/form-data/lib/browser.js"(exports2, module) {
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      module.exports = typeof self == "object" ? self.FormData : window.FormData;
    }
  });

  // src/index.ts
  var src_exports = {};
  __export(src_exports, {
    ApiError: () => ApiError,
    CancelError: () => CancelError,
    CancelablePromise: () => CancelablePromise,
    EventType: () => EventType,
    TabbyAgent: () => TabbyAgent,
    agentEventNames: () => agentEventNames
  });
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
  var kindOfTest = (type) => {
    type = type.toLowerCase();
    return (thing) => kindOf(thing) === type;
  };
  var typeOfTest = (type) => (thing) => typeof thing === type;
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
    let i2;
    let l2;
    if (typeof obj !== "object") {
      obj = [obj];
    }
    if (isArray(obj)) {
      for (i2 = 0, l2 = obj.length; i2 < l2; i2++) {
        fn.call(null, obj[i2], i2, obj);
      }
    } else {
      const keys = allOwnKeys ? Object.getOwnPropertyNames(obj) : Object.keys(obj);
      const len = keys.length;
      let key;
      for (i2 = 0; i2 < len; i2++) {
        key = keys[i2];
        fn.call(null, obj[key], key, obj);
      }
    }
  }
  function findKey(obj, key) {
    key = key.toLowerCase();
    const keys = Object.keys(obj);
    let i2 = keys.length;
    let _key;
    while (i2-- > 0) {
      _key = keys[i2];
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
    for (let i2 = 0, l2 = arguments.length; i2 < l2; i2++) {
      arguments[i2] && forEach(arguments[i2], assignValue);
    }
    return result;
  }
  var extend = (a2, b, thisArg, { allOwnKeys } = {}) => {
    forEach(b, (val, key) => {
      if (thisArg && isFunction(val)) {
        a2[key] = bind(val, thisArg);
      } else {
        a2[key] = val;
      }
    }, { allOwnKeys });
    return a2;
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
    let i2;
    let prop;
    const merged = {};
    destObj = destObj || {};
    if (sourceObj == null)
      return destObj;
    do {
      props = Object.getOwnPropertyNames(sourceObj);
      i2 = props.length;
      while (i2-- > 0) {
        prop = props[i2];
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
    let i2 = thing.length;
    if (!isNumber(i2))
      return null;
    const arr = new Array(i2);
    while (i2-- > 0) {
      arr[i2] = thing[i2];
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
      function replacer(m, p1, p2) {
        return p1.toUpperCase() + p2;
      }
    );
  };
  var hasOwnProperty = (({ hasOwnProperty: hasOwnProperty2 }) => (obj, prop) => hasOwnProperty2.call(obj, prop))(Object.prototype);
  var isRegExp = kindOfTest("RegExp");
  var reduceDescriptors = (obj, reducer) => {
    const descriptors2 = Object.getOwnPropertyDescriptors(obj);
    const reducedDescriptors = {};
    forEach(descriptors2, (descriptor, name2) => {
      if (reducer(descriptor, name2, obj) !== false) {
        reducedDescriptors[name2] = descriptor;
      }
    });
    Object.defineProperties(obj, reducedDescriptors);
  };
  var freezeMethods = (obj) => {
    reduceDescriptors(obj, (descriptor, name2) => {
      if (isFunction(obj) && ["arguments", "caller", "callee"].indexOf(name2) !== -1) {
        return false;
      }
      const value = obj[name2];
      if (!isFunction(value))
        return;
      descriptor.enumerable = false;
      if ("writable" in descriptor) {
        descriptor.writable = false;
        return;
      }
      if (!descriptor.set) {
        descriptor.set = () => {
          throw Error("Can not rewrite read-only method '" + name2 + "'");
        };
      }
    });
  };
  var toObjectSet = (arrayOrString, delimiter) => {
    const obj = {};
    const define2 = (arr) => {
      arr.forEach((value) => {
        obj[value] = true;
      });
    };
    isArray(arrayOrString) ? define2(arrayOrString) : define2(String(arrayOrString).split(delimiter));
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
    const visit = (source, i2) => {
      if (isObject(source)) {
        if (stack.indexOf(source) >= 0) {
          return;
        }
        if (!("toJSON" in source)) {
          stack[i2] = source;
          const target = isArray(source) ? [] : {};
          forEach(source, (value, key) => {
            const reducedValue = visit(value, i2 + 1);
            !isUndefined(reducedValue) && (target[key] = reducedValue);
          });
          stack[i2] = void 0;
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
  function renderKey(path, key, dots) {
    if (!path)
      return key;
    return path.concat(key).map(function each(token, i2) {
      token = removeBrackets(token);
      return !dots && i2 ? "[" + token + "]" : token;
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
    formData = formData || new (null_default || FormData)();
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
    function defaultVisitor(value, key, path) {
      let arr = value;
      if (value && !path && typeof value === "object") {
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
      formData.append(renderKey(path, key, dots), convertValue(value));
      return false;
    }
    const stack = [];
    const exposedHelpers = Object.assign(predicates, {
      defaultVisitor,
      convertValue,
      isVisitable
    });
    function build(value, path) {
      if (utils_default.isUndefined(value))
        return;
      if (stack.indexOf(value) !== -1) {
        throw Error("Circular reference detected in " + path.join("."));
      }
      stack.push(value);
      utils_default.forEach(value, function each(el, key) {
        const result = !(utils_default.isUndefined(el) || el === null) && visitor.call(
          formData,
          el,
          utils_default.isString(key) ? key.trim() : key,
          path,
          exposedHelpers
        );
        if (result === true) {
          build(el, path ? path.concat(key) : [key]);
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
  prototype2.append = function append(name2, value) {
    this._pairs.push([name2, value]);
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
      utils_default.forEach(this.handlers, function forEachHandler(h2) {
        if (h2 !== null) {
          fn(h2);
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
      visitor: function(value, key, path, helpers) {
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
  function parsePropPath(name2) {
    return utils_default.matchAll(/\w+|\[(\w*)]/g, name2).map((match) => {
      return match[0] === "[]" ? "" : match[1] || match[0];
    });
  }
  function arrayToObject(arr) {
    const obj = {};
    const keys = Object.keys(arr);
    let i2;
    const len = keys.length;
    let key;
    for (i2 = 0; i2 < len; i2++) {
      key = keys[i2];
      obj[key] = arr[key];
    }
    return obj;
  }
  function formDataToJSON(formData) {
    function buildPath(path, value, target, index) {
      let name2 = path[index++];
      const isNumericKey = Number.isFinite(+name2);
      const isLast = index >= path.length;
      name2 = !name2 && utils_default.isArray(target) ? target.length : name2;
      if (isLast) {
        if (utils_default.hasOwnProp(target, name2)) {
          target[name2] = [target[name2], value];
        } else {
          target[name2] = value;
        }
        return !isNumericKey;
      }
      if (!target[name2] || !utils_default.isObject(target[name2])) {
        target[name2] = [];
      }
      const result = buildPath(path, value, target[name2], index);
      if (result && utils_default.isArray(target[name2])) {
        target[name2] = arrayToObject(target[name2]);
      }
      return !isNumericKey;
    }
    if (utils_default.isFormData(formData) && utils_default.isFunction(formData.entries)) {
      const obj = {};
      utils_default.forEachEntry(formData, (name2, value) => {
        buildPath(parsePropPath(name2), value, obj, 0);
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
      } catch (e2) {
        if (e2.name !== "SyntaxError") {
          throw e2;
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
        } catch (e2) {
          if (strictJSONParsing) {
            if (e2.name === "SyntaxError") {
              throw AxiosError_default.from(e2, AxiosError_default.ERR_BAD_RESPONSE, this, null, this.response);
            }
            throw e2;
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
    let i2;
    rawHeaders && rawHeaders.split("\n").forEach(function parser(line) {
      i2 = line.indexOf(":");
      key = line.substring(0, i2).trim().toLowerCase();
      val = line.substring(i2 + 1).trim();
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
    return header.trim().toLowerCase().replace(/([a-z\d])(\w*)/g, (w, char, str) => {
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
      let i2 = keys.length;
      let deleted = false;
      while (i2--) {
        const key = keys[i2];
        if (!matcher || matchHeaderValue(this, this[key], key, matcher, true)) {
          delete this[key];
          deleted = true;
        }
      }
      return deleted;
    }
    normalize(format) {
      const self2 = this;
      const headers = {};
      utils_default.forEach(this, (value, header) => {
        const key = utils_default.findKey(headers, header);
        if (key) {
          self2[key] = normalizeValue(value);
          delete self2[header];
          return;
        }
        const normalized = format ? formatHeader(header) : String(header).trim();
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
  function settle(resolve2, reject, response) {
    const validateStatus2 = response.config.validateStatus;
    if (!response.status || !validateStatus2 || validateStatus2(response.status)) {
      resolve2(response);
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
        write: function write(name2, value, expires, path, domain2, secure) {
          const cookie = [];
          cookie.push(name2 + "=" + encodeURIComponent(value));
          if (utils_default.isNumber(expires)) {
            cookie.push("expires=" + new Date(expires).toGMTString());
          }
          if (utils_default.isString(path)) {
            cookie.push("path=" + path);
          }
          if (utils_default.isString(domain2)) {
            cookie.push("domain=" + domain2);
          }
          if (secure === true) {
            cookie.push("secure");
          }
          document.cookie = cookie.join("; ");
        },
        read: function read(name2) {
          const match = document.cookie.match(new RegExp("(^|;\\s*)(" + name2 + ")=([^;]*)"));
          return match ? decodeURIComponent(match[3]) : null;
        },
        remove: function remove(name2) {
          this.write(name2, "", Date.now() - 864e5);
        }
      };
    }()
  ) : (
    // Non standard browser env (web workers, react-native) lack needed support.
    function nonStandardBrowserEnv() {
      return {
        write: function write() {
        },
        read: function read() {
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
      let i2 = tail;
      let bytesCount = 0;
      while (i2 !== head) {
        bytesCount += bytes[i2++];
        i2 = i2 % samplesCount;
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
    return (e2) => {
      const loaded = e2.loaded;
      const total = e2.lengthComputable ? e2.total : void 0;
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
        event: e2
      };
      data[isDownloadStream ? "download" : "upload"] = true;
      listener(data);
    };
  }
  var isXHRAdapterSupported = typeof XMLHttpRequest !== "undefined";
  var xhr_default = isXHRAdapterSupported && function(config2) {
    return new Promise(function dispatchXhrRequest(resolve2, reject) {
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
          resolve2(value);
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
      } catch (e2) {
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
      for (let i2 = 0; i2 < length; i2++) {
        nameOrAdapter = adapters[i2];
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
    function mergeDeepProperties(a2, b, caseless) {
      if (!utils_default.isUndefined(b)) {
        return getMergedValue(a2, b, caseless);
      } else if (!utils_default.isUndefined(a2)) {
        return getMergedValue(void 0, a2, caseless);
      }
    }
    function valueFromConfig2(a2, b) {
      if (!utils_default.isUndefined(b)) {
        return getMergedValue(void 0, b);
      }
    }
    function defaultToConfig2(a2, b) {
      if (!utils_default.isUndefined(b)) {
        return getMergedValue(void 0, b);
      } else if (!utils_default.isUndefined(a2)) {
        return getMergedValue(void 0, a2);
      }
    }
    function mergeDirectKeys(a2, b, prop) {
      if (prop in config2) {
        return getMergedValue(a2, b);
      } else if (prop in config1) {
        return getMergedValue(void 0, a2);
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
      headers: (a2, b) => mergeDeepProperties(headersToObject(a2), headersToObject(b), true)
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
  ["object", "boolean", "number", "function", "string", "symbol"].forEach((type, i2) => {
    validators[type] = function validator(thing) {
      return typeof thing === type || "a" + (i2 < 1 ? "n " : " ") + type;
    };
  });
  var deprecatedWarnings = {};
  validators.transitional = function transitional(validator, version2, message) {
    function formatMessage(opt, desc) {
      return "[Axios v" + VERSION + "] Transitional option '" + opt + "'" + desc + (message ? ". " + message : "");
    }
    return (value, opt, opts) => {
      if (validator === false) {
        throw new AxiosError_default(
          formatMessage(opt, " has been removed" + (version2 ? " in " + version2 : "")),
          AxiosError_default.ERR_DEPRECATED
        );
      }
      if (version2 && !deprecatedWarnings[opt]) {
        deprecatedWarnings[opt] = true;
        console.warn(
          formatMessage(
            opt,
            " has been deprecated since v" + version2 + " and will be removed in the near future"
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
    let i2 = keys.length;
    while (i2-- > 0) {
      const opt = keys[i2];
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
      let i2 = 0;
      let len;
      if (!synchronousRequestInterceptors) {
        const chain = [dispatchRequest.bind(this), void 0];
        chain.unshift.apply(chain, requestInterceptorChain);
        chain.push.apply(chain, responseInterceptorChain);
        len = chain.length;
        promise = Promise.resolve(config2);
        while (i2 < len) {
          promise = promise.then(chain[i2++], chain[i2++]);
        }
        return promise;
      }
      len = requestInterceptorChain.length;
      let newConfig = config2;
      i2 = 0;
      while (i2 < len) {
        const onFulfilled = requestInterceptorChain[i2++];
        const onRejected = requestInterceptorChain[i2++];
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
      i2 = 0;
      len = responseInterceptorChain.length;
      while (i2 < len) {
        promise = promise.then(responseInterceptorChain[i2++], responseInterceptorChain[i2++]);
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
      this.promise = new Promise(function promiseExecutor(resolve2) {
        resolvePromise = resolve2;
      });
      const token = this;
      this.promise.then((cancel) => {
        if (!token._listeners)
          return;
        let i2 = token._listeners.length;
        while (i2-- > 0) {
          token._listeners[i2](cancel);
        }
        token._listeners = null;
      });
      this.promise.then = (onfulfilled) => {
        let _resolve2;
        const promise = new Promise((resolve2) => {
          token.subscribe(resolve2);
          _resolve2 = resolve2;
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
      const token = new CancelToken(function executor(c2) {
        cancel = c2;
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
  axios.all = function all(promises) {
    return Promise.all(promises);
  };
  axios.spread = spread;
  axios.isAxiosError = isAxiosError;
  axios.mergeConfig = mergeConfig;
  axios.AxiosHeaders = AxiosHeaders_default;
  axios.formToJSON = (thing) => formDataToJSON_default(utils_default.isHTMLForm(thing) ? new FormData(thing) : thing);
  axios.HttpStatusCode = HttpStatusCode_default;
  axios.default = axios;
  var axios_default = axios;

  // node_modules/axios/index.js
  var {
    Axios: Axios2,
    AxiosError: AxiosError2,
    CanceledError: CanceledError2,
    isCancel: isCancel2,
    CancelToken: CancelToken2,
    VERSION: VERSION2,
    all: all2,
    Cancel,
    isAxiosError: isAxiosError2,
    spread: spread2,
    toFormData: toFormData2,
    AxiosHeaders: AxiosHeaders2,
    HttpStatusCode: HttpStatusCode2,
    formToJSON,
    mergeConfig: mergeConfig2
  } = axios_default;

  // node_modules/@jspm/core/nodelibs/browser/events.js
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();

  // node_modules/@jspm/core/nodelibs/browser/chunk-4bd36a8f.js
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  var e;
  var t;
  var n = "object" == typeof Reflect ? Reflect : null;
  var r = n && "function" == typeof n.apply ? n.apply : function(e2, t2, n2) {
    return Function.prototype.apply.call(e2, t2, n2);
  };
  t = n && "function" == typeof n.ownKeys ? n.ownKeys : Object.getOwnPropertySymbols ? function(e2) {
    return Object.getOwnPropertyNames(e2).concat(Object.getOwnPropertySymbols(e2));
  } : function(e2) {
    return Object.getOwnPropertyNames(e2);
  };
  var i = Number.isNaN || function(e2) {
    return e2 != e2;
  };
  function o() {
    o.init.call(this);
  }
  e = o, o.EventEmitter = o, o.prototype._events = void 0, o.prototype._eventsCount = 0, o.prototype._maxListeners = void 0;
  var s = 10;
  function u(e2) {
    if ("function" != typeof e2)
      throw new TypeError('The "listener" argument must be of type Function. Received type ' + typeof e2);
  }
  function f(e2) {
    return void 0 === e2._maxListeners ? o.defaultMaxListeners : e2._maxListeners;
  }
  function v(e2, t2, n2, r2) {
    var i2, o2, s2, v2;
    if (u(n2), void 0 === (o2 = e2._events) ? (o2 = e2._events = /* @__PURE__ */ Object.create(null), e2._eventsCount = 0) : (void 0 !== o2.newListener && (e2.emit("newListener", t2, n2.listener ? n2.listener : n2), o2 = e2._events), s2 = o2[t2]), void 0 === s2)
      s2 = o2[t2] = n2, ++e2._eventsCount;
    else if ("function" == typeof s2 ? s2 = o2[t2] = r2 ? [n2, s2] : [s2, n2] : r2 ? s2.unshift(n2) : s2.push(n2), (i2 = f(e2)) > 0 && s2.length > i2 && !s2.warned) {
      s2.warned = true;
      var a2 = new Error("Possible EventEmitter memory leak detected. " + s2.length + " " + String(t2) + " listeners added. Use emitter.setMaxListeners() to increase limit");
      a2.name = "MaxListenersExceededWarning", a2.emitter = e2, a2.type = t2, a2.count = s2.length, v2 = a2, console && console.warn && console.warn(v2);
    }
    return e2;
  }
  function a() {
    if (!this.fired)
      return this.target.removeListener(this.type, this.wrapFn), this.fired = true, 0 === arguments.length ? this.listener.call(this.target) : this.listener.apply(this.target, arguments);
  }
  function l(e2, t2, n2) {
    var r2 = { fired: false, wrapFn: void 0, target: e2, type: t2, listener: n2 }, i2 = a.bind(r2);
    return i2.listener = n2, r2.wrapFn = i2, i2;
  }
  function h(e2, t2, n2) {
    var r2 = e2._events;
    if (void 0 === r2)
      return [];
    var i2 = r2[t2];
    return void 0 === i2 ? [] : "function" == typeof i2 ? n2 ? [i2.listener || i2] : [i2] : n2 ? function(e3) {
      for (var t3 = new Array(e3.length), n3 = 0; n3 < t3.length; ++n3)
        t3[n3] = e3[n3].listener || e3[n3];
      return t3;
    }(i2) : c(i2, i2.length);
  }
  function p(e2) {
    var t2 = this._events;
    if (void 0 !== t2) {
      var n2 = t2[e2];
      if ("function" == typeof n2)
        return 1;
      if (void 0 !== n2)
        return n2.length;
    }
    return 0;
  }
  function c(e2, t2) {
    for (var n2 = new Array(t2), r2 = 0; r2 < t2; ++r2)
      n2[r2] = e2[r2];
    return n2;
  }
  Object.defineProperty(o, "defaultMaxListeners", { enumerable: true, get: function() {
    return s;
  }, set: function(e2) {
    if ("number" != typeof e2 || e2 < 0 || i(e2))
      throw new RangeError('The value of "defaultMaxListeners" is out of range. It must be a non-negative number. Received ' + e2 + ".");
    s = e2;
  } }), o.init = function() {
    void 0 !== this._events && this._events !== Object.getPrototypeOf(this)._events || (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0), this._maxListeners = this._maxListeners || void 0;
  }, o.prototype.setMaxListeners = function(e2) {
    if ("number" != typeof e2 || e2 < 0 || i(e2))
      throw new RangeError('The value of "n" is out of range. It must be a non-negative number. Received ' + e2 + ".");
    return this._maxListeners = e2, this;
  }, o.prototype.getMaxListeners = function() {
    return f(this);
  }, o.prototype.emit = function(e2) {
    for (var t2 = [], n2 = 1; n2 < arguments.length; n2++)
      t2.push(arguments[n2]);
    var i2 = "error" === e2, o2 = this._events;
    if (void 0 !== o2)
      i2 = i2 && void 0 === o2.error;
    else if (!i2)
      return false;
    if (i2) {
      var s2;
      if (t2.length > 0 && (s2 = t2[0]), s2 instanceof Error)
        throw s2;
      var u2 = new Error("Unhandled error." + (s2 ? " (" + s2.message + ")" : ""));
      throw u2.context = s2, u2;
    }
    var f2 = o2[e2];
    if (void 0 === f2)
      return false;
    if ("function" == typeof f2)
      r(f2, this, t2);
    else {
      var v2 = f2.length, a2 = c(f2, v2);
      for (n2 = 0; n2 < v2; ++n2)
        r(a2[n2], this, t2);
    }
    return true;
  }, o.prototype.addListener = function(e2, t2) {
    return v(this, e2, t2, false);
  }, o.prototype.on = o.prototype.addListener, o.prototype.prependListener = function(e2, t2) {
    return v(this, e2, t2, true);
  }, o.prototype.once = function(e2, t2) {
    return u(t2), this.on(e2, l(this, e2, t2)), this;
  }, o.prototype.prependOnceListener = function(e2, t2) {
    return u(t2), this.prependListener(e2, l(this, e2, t2)), this;
  }, o.prototype.removeListener = function(e2, t2) {
    var n2, r2, i2, o2, s2;
    if (u(t2), void 0 === (r2 = this._events))
      return this;
    if (void 0 === (n2 = r2[e2]))
      return this;
    if (n2 === t2 || n2.listener === t2)
      0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : (delete r2[e2], r2.removeListener && this.emit("removeListener", e2, n2.listener || t2));
    else if ("function" != typeof n2) {
      for (i2 = -1, o2 = n2.length - 1; o2 >= 0; o2--)
        if (n2[o2] === t2 || n2[o2].listener === t2) {
          s2 = n2[o2].listener, i2 = o2;
          break;
        }
      if (i2 < 0)
        return this;
      0 === i2 ? n2.shift() : !function(e3, t3) {
        for (; t3 + 1 < e3.length; t3++)
          e3[t3] = e3[t3 + 1];
        e3.pop();
      }(n2, i2), 1 === n2.length && (r2[e2] = n2[0]), void 0 !== r2.removeListener && this.emit("removeListener", e2, s2 || t2);
    }
    return this;
  }, o.prototype.off = o.prototype.removeListener, o.prototype.removeAllListeners = function(e2) {
    var t2, n2, r2;
    if (void 0 === (n2 = this._events))
      return this;
    if (void 0 === n2.removeListener)
      return 0 === arguments.length ? (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0) : void 0 !== n2[e2] && (0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : delete n2[e2]), this;
    if (0 === arguments.length) {
      var i2, o2 = Object.keys(n2);
      for (r2 = 0; r2 < o2.length; ++r2)
        "removeListener" !== (i2 = o2[r2]) && this.removeAllListeners(i2);
      return this.removeAllListeners("removeListener"), this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0, this;
    }
    if ("function" == typeof (t2 = n2[e2]))
      this.removeListener(e2, t2);
    else if (void 0 !== t2)
      for (r2 = t2.length - 1; r2 >= 0; r2--)
        this.removeListener(e2, t2[r2]);
    return this;
  }, o.prototype.listeners = function(e2) {
    return h(this, e2, true);
  }, o.prototype.rawListeners = function(e2) {
    return h(this, e2, false);
  }, o.listenerCount = function(e2, t2) {
    return "function" == typeof e2.listenerCount ? e2.listenerCount(t2) : p.call(e2, t2);
  }, o.prototype.listenerCount = p, o.prototype.eventNames = function() {
    return this._eventsCount > 0 ? t(this._events) : [];
  };
  var y = e;
  y.EventEmitter;
  y.defaultMaxListeners;
  y.init;
  y.listenerCount;
  y.EventEmitter;
  y.defaultMaxListeners;
  y.init;
  y.listenerCount;

  // node_modules/@jspm/core/nodelibs/browser/events.js
  y.once = function(emitter, event) {
    return new Promise((resolve2, reject) => {
      function eventListener(...args) {
        if (errorListener !== void 0) {
          emitter.removeListener("error", errorListener);
        }
        resolve2(args);
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
    let finished = false;
    const iterator = {
      async next() {
        const value = unconsumedEventValues.shift();
        if (value) {
          return createIterResult(value, false);
        }
        if (error) {
          const p2 = Promise.reject(error);
          error = null;
          return p2;
        }
        if (finished) {
          return createIterResult(void 0, true);
        }
        return new Promise((resolve2, reject) => unconsumedPromises.push({ resolve: resolve2, reject }));
      },
      async return() {
        emitter.removeListener(event, eventHandler);
        emitter.removeListener("error", errorHandler);
        finished = true;
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
      finished = true;
      const toError = unconsumedPromises.shift();
      if (toError) {
        toError.reject(err);
      } else {
        error = err;
      }
      iterator.return();
    }
  };
  var {
    EventEmitter,
    defaultMaxListeners,
    init,
    listenerCount,
    on: on2,
    once: once2
  } = y;

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
  for (let i2 = 0; i2 < 256; ++i2) {
    byteToHex.push((i2 + 256).toString(16).slice(1));
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
      for (let i2 = 0; i2 < 16; ++i2) {
        buf[offset + i2] = rnds[i2];
      }
      return buf;
    }
    return unsafeStringify(rnds);
  }
  var v4_default = v4;

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
  var emitWarning2 = (msg, type, code, fn) => {
    typeof PROCESS.emitWarning === "function" ? PROCESS.emitWarning(msg, type, code, fn) : console.error(`[${code}] ${type}: ${msg}`);
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
      addEventListener(_, fn) {
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
  var TYPE = Symbol("type");
  var isPosInt = (n2) => n2 && n2 === Math.floor(n2) && n2 > 0 && isFinite(n2);
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
      const s2 = new _Stack(max, HeapCls);
      __privateSet(_Stack, _constructing, false);
      return s2;
    }
    push(n2) {
      this.heap[this.length++] = n2;
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
    static unsafeExposeInternals(c2) {
      return {
        // properties
        starts: __privateGet(c2, _starts),
        ttls: __privateGet(c2, _ttls),
        sizes: __privateGet(c2, _sizes),
        keyMap: __privateGet(c2, _keyMap),
        keyList: __privateGet(c2, _keyList),
        valList: __privateGet(c2, _valList),
        next: __privateGet(c2, _next),
        prev: __privateGet(c2, _prev),
        get head() {
          return __privateGet(c2, _head);
        },
        get tail() {
          return __privateGet(c2, _tail);
        },
        free: __privateGet(c2, _free),
        // methods
        isBackgroundFetch: (p2) => {
          var _a;
          return __privateMethod(_a = c2, _isBackgroundFetch, isBackgroundFetch_fn).call(_a, p2);
        },
        backgroundFetch: (k, index, options, context) => {
          var _a;
          return __privateMethod(_a = c2, _backgroundFetch, backgroundFetch_fn).call(_a, k, index, options, context);
        },
        moveToTail: (index) => {
          var _a;
          return __privateMethod(_a = c2, _moveToTail, moveToTail_fn).call(_a, index);
        },
        indexes: (options) => {
          var _a;
          return __privateMethod(_a = c2, _indexes, indexes_fn).call(_a, options);
        },
        rindexes: (options) => {
          var _a;
          return __privateMethod(_a = c2, _rindexes, rindexes_fn).call(_a, options);
        },
        isStale: (index) => {
          var _a;
          return __privateGet(_a = c2, _isStale).call(_a, index);
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
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
        if (__privateGet(this, _valList)[i2] !== void 0 && __privateGet(this, _keyList)[i2] !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield [__privateGet(this, _keyList)[i2], __privateGet(this, _valList)[i2]];
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
      for (const i2 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
        if (__privateGet(this, _valList)[i2] !== void 0 && __privateGet(this, _keyList)[i2] !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield [__privateGet(this, _keyList)[i2], __privateGet(this, _valList)[i2]];
        }
      }
    }
    /**
     * Return a generator yielding the keys in the cache,
     * in order from most recently used to least recently used.
     */
    *keys() {
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
        const k = __privateGet(this, _keyList)[i2];
        if (k !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield k;
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
      for (const i2 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
        const k = __privateGet(this, _keyList)[i2];
        if (k !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield k;
        }
      }
    }
    /**
     * Return a generator yielding the values in the cache,
     * in order from most recently used to least recently used.
     */
    *values() {
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
        const v2 = __privateGet(this, _valList)[i2];
        if (v2 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield __privateGet(this, _valList)[i2];
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
      for (const i2 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
        const v2 = __privateGet(this, _valList)[i2];
        if (v2 !== void 0 && !__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, __privateGet(this, _valList)[i2])) {
          yield __privateGet(this, _valList)[i2];
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
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
        const v2 = __privateGet(this, _valList)[i2];
        const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) ? v2.__staleWhileFetching : v2;
        if (value === void 0)
          continue;
        if (fn(value, __privateGet(this, _keyList)[i2], this)) {
          return this.get(__privateGet(this, _keyList)[i2], getOptions);
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
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this)) {
        const v2 = __privateGet(this, _valList)[i2];
        const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) ? v2.__staleWhileFetching : v2;
        if (value === void 0)
          continue;
        fn.call(thisp, value, __privateGet(this, _keyList)[i2], this);
      }
    }
    /**
     * The same as {@link LRUCache.forEach} but items are iterated over in
     * reverse order.  (ie, less recently used items are iterated over first.)
     */
    rforEach(fn, thisp = this) {
      for (const i2 of __privateMethod(this, _rindexes, rindexes_fn).call(this)) {
        const v2 = __privateGet(this, _valList)[i2];
        const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) ? v2.__staleWhileFetching : v2;
        if (value === void 0)
          continue;
        fn.call(thisp, value, __privateGet(this, _keyList)[i2], this);
      }
    }
    /**
     * Delete any stale entries. Returns true if anything was removed,
     * false otherwise.
     */
    purgeStale() {
      let deleted = false;
      for (const i2 of __privateMethod(this, _rindexes, rindexes_fn).call(this, { allowStale: true })) {
        if (__privateGet(this, _isStale).call(this, i2)) {
          this.delete(__privateGet(this, _keyList)[i2]);
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
      for (const i2 of __privateMethod(this, _indexes, indexes_fn).call(this, { allowStale: true })) {
        const key = __privateGet(this, _keyList)[i2];
        const v2 = __privateGet(this, _valList)[i2];
        const value = __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) ? v2.__staleWhileFetching : v2;
        if (value === void 0 || key === void 0)
          continue;
        const entry = { value };
        if (__privateGet(this, _ttls) && __privateGet(this, _starts)) {
          entry.ttl = __privateGet(this, _ttls)[i2];
          const age = perf.now() - __privateGet(this, _starts)[i2];
          entry.start = Math.floor(Date.now() - age);
        }
        if (__privateGet(this, _sizes)) {
          entry.size = __privateGet(this, _sizes)[i2];
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
    set(k, v2, setOptions = {}) {
      var _a, _b;
      if (v2 === void 0) {
        this.delete(k);
        return this;
      }
      const { ttl = this.ttl, start, noDisposeOnSet = this.noDisposeOnSet, sizeCalculation = this.sizeCalculation, status } = setOptions;
      let { noUpdateTTL = this.noUpdateTTL } = setOptions;
      const size = __privateGet(this, _requireSize).call(this, k, v2, setOptions.size || 0, sizeCalculation);
      if (this.maxEntrySize && size > this.maxEntrySize) {
        if (status) {
          status.set = "miss";
          status.maxEntrySizeExceeded = true;
        }
        this.delete(k);
        return this;
      }
      let index = __privateGet(this, _size) === 0 ? void 0 : __privateGet(this, _keyMap).get(k);
      if (index === void 0) {
        index = __privateGet(this, _size) === 0 ? __privateGet(this, _tail) : __privateGet(this, _free).length !== 0 ? __privateGet(this, _free).pop() : __privateGet(this, _size) === __privateGet(this, _max) ? __privateMethod(this, _evict, evict_fn).call(this, false) : __privateGet(this, _size);
        __privateGet(this, _keyList)[index] = k;
        __privateGet(this, _valList)[index] = v2;
        __privateGet(this, _keyMap).set(k, index);
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
        if (v2 !== oldVal) {
          if (__privateGet(this, _hasFetchMethod) && __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, oldVal)) {
            oldVal.__abortController.abort(new Error("replaced"));
          } else if (!noDisposeOnSet) {
            if (__privateGet(this, _hasDispose)) {
              (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, oldVal, k, "set");
            }
            if (__privateGet(this, _hasDisposeAfter)) {
              __privateGet(this, _disposed)?.push([oldVal, k, "set"]);
            }
          }
          __privateGet(this, _removeItemSize).call(this, index);
          __privateGet(this, _addItemSize).call(this, index, size, status);
          __privateGet(this, _valList)[index] = v2;
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
    has(k, hasOptions = {}) {
      const { updateAgeOnHas = this.updateAgeOnHas, status } = hasOptions;
      const index = __privateGet(this, _keyMap).get(k);
      if (index !== void 0) {
        const v2 = __privateGet(this, _valList)[index];
        if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) && v2.__staleWhileFetching === void 0) {
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
    peek(k, peekOptions = {}) {
      const { allowStale = this.allowStale } = peekOptions;
      const index = __privateGet(this, _keyMap).get(k);
      if (index !== void 0 && (allowStale || !__privateGet(this, _isStale).call(this, index))) {
        const v2 = __privateGet(this, _valList)[index];
        return __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2) ? v2.__staleWhileFetching : v2;
      }
    }
    async fetch(k, fetchOptions = {}) {
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
        return this.get(k, {
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
      let index = __privateGet(this, _keyMap).get(k);
      if (index === void 0) {
        if (status)
          status.fetch = "miss";
        const p2 = __privateMethod(this, _backgroundFetch, backgroundFetch_fn).call(this, k, index, options, context);
        return p2.__returned = p2;
      } else {
        const v2 = __privateGet(this, _valList)[index];
        if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
          const stale = allowStale && v2.__staleWhileFetching !== void 0;
          if (status) {
            status.fetch = "inflight";
            if (stale)
              status.returnedStale = true;
          }
          return stale ? v2.__staleWhileFetching : v2.__returned = v2;
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
          return v2;
        }
        const p2 = __privateMethod(this, _backgroundFetch, backgroundFetch_fn).call(this, k, index, options, context);
        const hasStale = p2.__staleWhileFetching !== void 0;
        const staleVal = hasStale && allowStale;
        if (status) {
          status.fetch = isStale ? "stale" : "refresh";
          if (staleVal && isStale)
            status.returnedStale = true;
        }
        return staleVal ? p2.__staleWhileFetching : p2.__returned = p2;
      }
    }
    /**
     * Return a value from the cache. Will update the recency of the cache
     * entry found.
     *
     * If the key is not found, get() will return `undefined`.
     */
    get(k, getOptions = {}) {
      const { allowStale = this.allowStale, updateAgeOnGet = this.updateAgeOnGet, noDeleteOnStaleGet = this.noDeleteOnStaleGet, status } = getOptions;
      const index = __privateGet(this, _keyMap).get(k);
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
              this.delete(k);
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
    delete(k) {
      var _a, _b;
      let deleted = false;
      if (__privateGet(this, _size) !== 0) {
        const index = __privateGet(this, _keyMap).get(k);
        if (index !== void 0) {
          deleted = true;
          if (__privateGet(this, _size) === 1) {
            this.clear();
          } else {
            __privateGet(this, _removeItemSize).call(this, index);
            const v2 = __privateGet(this, _valList)[index];
            if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
              v2.__abortController.abort(new Error("deleted"));
            } else if (__privateGet(this, _hasDispose) || __privateGet(this, _hasDisposeAfter)) {
              if (__privateGet(this, _hasDispose)) {
                (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v2, k, "delete");
              }
              if (__privateGet(this, _hasDisposeAfter)) {
                __privateGet(this, _disposed)?.push([v2, k, "delete"]);
              }
            }
            __privateGet(this, _keyMap).delete(k);
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
        const v2 = __privateGet(this, _valList)[index];
        if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
          v2.__abortController.abort(new Error("deleted"));
        } else {
          const k = __privateGet(this, _keyList)[index];
          if (__privateGet(this, _hasDispose)) {
            (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v2, k, "delete");
          }
          if (__privateGet(this, _hasDisposeAfter)) {
            __privateGet(this, _disposed)?.push([v2, k, "delete"]);
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
        const t2 = setTimeout(() => {
          if (__privateGet(this, _isStale).call(this, index)) {
            this.delete(__privateGet(this, _keyList)[index]);
          }
        }, ttl + 1);
        if (t2.unref) {
          t2.unref();
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
      const n2 = perf.now();
      if (this.ttlResolution > 0) {
        cachedNow = n2;
        const t2 = setTimeout(() => cachedNow = 0, this.ttlResolution);
        if (t2.unref) {
          t2.unref();
        }
      }
      return n2;
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
    __privateSet(this, _requireSize, (k, v2, size, sizeCalculation) => {
      if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
        return 0;
      }
      if (!isPosInt(size)) {
        if (sizeCalculation) {
          if (typeof sizeCalculation !== "function") {
            throw new TypeError("sizeCalculation must be a function");
          }
          size = sizeCalculation(v2, k);
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
      for (let i2 = __privateGet(this, _tail); true; ) {
        if (!__privateMethod(this, _isValidIndex, isValidIndex_fn).call(this, i2)) {
          break;
        }
        if (allowStale || !__privateGet(this, _isStale).call(this, i2)) {
          yield i2;
        }
        if (i2 === __privateGet(this, _head)) {
          break;
        } else {
          i2 = __privateGet(this, _prev)[i2];
        }
      }
    }
  };
  _rindexes = new WeakSet();
  rindexes_fn = function* ({ allowStale = this.allowStale } = {}) {
    if (__privateGet(this, _size)) {
      for (let i2 = __privateGet(this, _head); true; ) {
        if (!__privateMethod(this, _isValidIndex, isValidIndex_fn).call(this, i2)) {
          break;
        }
        if (allowStale || !__privateGet(this, _isStale).call(this, i2)) {
          yield i2;
        }
        if (i2 === __privateGet(this, _tail)) {
          break;
        } else {
          i2 = __privateGet(this, _next)[i2];
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
    const k = __privateGet(this, _keyList)[head];
    const v2 = __privateGet(this, _valList)[head];
    if (__privateGet(this, _hasFetchMethod) && __privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
      v2.__abortController.abort(new Error("evicted"));
    } else if (__privateGet(this, _hasDispose) || __privateGet(this, _hasDisposeAfter)) {
      if (__privateGet(this, _hasDispose)) {
        (_a = __privateGet(this, _dispose)) == null ? void 0 : _a.call(this, v2, k, "evict");
      }
      if (__privateGet(this, _hasDisposeAfter)) {
        __privateGet(this, _disposed)?.push([v2, k, "evict"]);
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
    __privateGet(this, _keyMap).delete(k);
    __privateWrapper(this, _size)._--;
    return head;
  };
  _backgroundFetch = new WeakSet();
  backgroundFetch_fn = function(k, index, options, context) {
    const v2 = index === void 0 ? void 0 : __privateGet(this, _valList)[index];
    if (__privateMethod(this, _isBackgroundFetch, isBackgroundFetch_fn).call(this, v2)) {
      return v2;
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
    const cb = (v3, updateCache = false) => {
      const { aborted } = ac.signal;
      const ignoreAbort = options.ignoreFetchAbort && v3 !== void 0;
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
      const bf2 = p2;
      if (__privateGet(this, _valList)[index] === p2) {
        if (v3 === void 0) {
          if (bf2.__staleWhileFetching) {
            __privateGet(this, _valList)[index] = bf2.__staleWhileFetching;
          } else {
            this.delete(k);
          }
        } else {
          if (options.status)
            options.status.fetchUpdated = true;
          this.set(k, v3, fetchOpts.options);
        }
      }
      return v3;
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
      const bf2 = p2;
      if (__privateGet(this, _valList)[index] === p2) {
        const del = !noDelete || bf2.__staleWhileFetching === void 0;
        if (del) {
          this.delete(k);
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
      const fmp = (_a = __privateGet(this, _fetchMethod)) == null ? void 0 : _a.call(this, k, v2, fetchOpts);
      if (fmp && fmp instanceof Promise) {
        fmp.then((v3) => res(v3), rej);
      }
      ac.signal.addEventListener("abort", () => {
        if (!options.ignoreFetchAbort || options.allowStaleOnFetchAbort) {
          res();
          if (options.allowStaleOnFetchAbort) {
            res = (v3) => cb(v3, true);
          }
        }
      });
    };
    if (options.status)
      options.status.fetchDispatched = true;
    const p2 = new Promise(pcall).then(cb, eb);
    const bf = Object.assign(p2, {
      __abortController: ac,
      __staleWhileFetching: v2,
      __returned: void 0
    });
    if (index === void 0) {
      this.set(k, bf, { ...fetchOpts.options, status: void 0 });
      index = __privateGet(this, _keyMap).get(k);
    } else {
      __privateGet(this, _valList)[index] = bf;
    }
    return bf;
  };
  _isBackgroundFetch = new WeakSet();
  isBackgroundFetch_fn = function(p2) {
    if (!__privateGet(this, _hasFetchMethod))
      return false;
    const b = p2;
    return !!b && b instanceof Promise && b.hasOwnProperty("__staleWhileFetching") && b.__abortController instanceof AC;
  };
  _connect = new WeakSet();
  connect_fn = function(p2, n2) {
    __privateGet(this, _prev)[n2] = p2;
    __privateGet(this, _next)[p2] = n2;
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

  // src/utils.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();

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
      __privateSet(this, _promise, new Promise((resolve2, reject) => {
        __privateSet(this, _resolve, resolve2);
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
    const process2 = (key, value) => {
      if (isDefined(value)) {
        if (Array.isArray(value)) {
          value.forEach((v2) => {
            process2(key, v2);
          });
        } else if (typeof value === "object") {
          Object.entries(value).forEach(([k, v2]) => {
            process2(`${key}[${k}]`, v2);
          });
        } else {
          append2(key, value);
        }
      }
    };
    Object.entries(params).forEach(([key, value]) => {
      process2(key, value);
    });
    if (qs.length > 0) {
      return `?${qs.join("&")}`;
    }
    return "";
  };
  var getUrl = (config2, options) => {
    const encoder = config2.ENCODE_PATH || encodeURI;
    const path = options.url.replace("{api-version}", config2.VERSION).replace(/{(.*?)}/g, (substring, group) => {
      if (options.path?.hasOwnProperty(group)) {
        return encoder(String(options.path[group]));
      }
      return substring;
    });
    const url = `${config2.BASE}${path}`;
    if (options.query) {
      return `${url}${getQueryString(options.query)}`;
    }
    return url;
  };
  var getFormData = (options) => {
    if (options.formData) {
      const formData = new import_form_data.default();
      const process2 = (key, value) => {
        if (isString2(value) || isBlob2(value)) {
          formData.append(key, value);
        } else {
          formData.append(key, JSON.stringify(value));
        }
      };
      Object.entries(options.formData).filter(([_, value]) => isDefined(value)).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          value.forEach((v2) => process2(key, v2));
        } else {
          process2(key, value);
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
    }).filter(([_, value]) => isDefined(value)).reduce((headers2, [key, value]) => ({
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
    return new CancelablePromise(async (resolve2, reject, onCancel) => {
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
          resolve2(result.body);
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

  // src/generated/services/DefaultService.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  var DefaultService = class {
    constructor(httpRequest) {
      this.httpRequest = httpRequest;
    }
    /**
     * Completions
     * @param requestBody
     * @returns CompletionResponse Successful Response
     * @throws ApiError
     */
    completionsV1CompletionsPost(requestBody) {
      return this.httpRequest.request({
        method: "POST",
        url: "/v1/completions",
        body: requestBody,
        mediaType: "application/json",
        errors: {
          422: `Validation Error`
        }
      });
    }
    /**
     * Events
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    eventsV1EventsPost(requestBody) {
      return this.httpRequest.request({
        method: "POST",
        url: "/v1/events",
        body: requestBody,
        mediaType: "application/json",
        errors: {
          422: `Validation Error`
        }
      });
    }
  };

  // src/generated/TabbyApi.ts
  var TabbyApi = class {
    constructor(config2, HttpRequest = AxiosHttpRequest) {
      this.request = new HttpRequest({
        BASE: config2?.BASE ?? "",
        VERSION: config2?.VERSION ?? "0.1.0",
        WITH_CREDENTIALS: config2?.WITH_CREDENTIALS ?? false,
        CREDENTIALS: config2?.CREDENTIALS ?? "include",
        TOKEN: config2?.TOKEN,
        USERNAME: config2?.USERNAME,
        PASSWORD: config2?.PASSWORD,
        HEADERS: config2?.HEADERS,
        ENCODE_PATH: config2?.ENCODE_PATH
      });
      this.default = new DefaultService(this.request);
    }
  };

  // src/generated/core/OpenAPI.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();

  // src/generated/models/EventType.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  var EventType = /* @__PURE__ */ ((EventType2) => {
    EventType2["COMPLETION"] = "completion";
    EventType2["VIEW"] = "view";
    EventType2["SELECT"] = "select";
    return EventType2;
  })(EventType || {});

  // src/utils.ts
  function sleep(milliseconds) {
    return new Promise((r2) => setTimeout(r2, milliseconds));
  }
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
    return new CancelablePromise((resolve2, reject, onCancel) => {
      promise.then((resp) => {
        resolve2(resp);
      }).catch((err) => {
        reject(err);
      });
      onCancel(() => {
        cancel();
      });
    });
  }

  // src/CompletionCache.ts
  var CompletionCache = class {
    constructor() {
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
        this.cache.set(this.hash(entry.key), entry.value);
      }
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
      return positions.filter((v2, i2, arr) => arr.indexOf(v2) === i2).sort((a2, b) => a2 - b);
    }
  };

  // src/TabbyAgent.ts
  var TabbyAgent = class extends EventEmitter {
    constructor() {
      super();
      this.serverUrl = "http://127.0.0.1:5000";
      this.status = "connecting";
      this.ping();
      this.api = new TabbyApi({ BASE: this.serverUrl });
      this.completionCache = new CompletionCache();
    }
    changeStatus(status) {
      if (this.status != status) {
        this.status = status;
        const event = { event: "statusChanged", status };
        super.emit("statusChanged", event);
      }
    }
    async ping(tries = 0) {
      try {
        const response = await axios_default.get(this.serverUrl);
        this.changeStatus("ready");
        return true;
      } catch (e2) {
        if (tries > 5) {
          this.changeStatus("disconnected");
          return false;
        }
        this.changeStatus("connecting");
        const pingRetryDelay = 1e3;
        await sleep(pingRetryDelay);
        return this.ping(tries + 1);
      }
    }
    wrapApiPromise(promise) {
      return cancelable(
        promise.then((resolved) => {
          this.changeStatus("ready");
          return resolved;
        }).catch((err) => {
          this.changeStatus("disconnected");
          throw err;
        }),
        () => {
          promise.cancel();
        }
      );
    }
    createPrompt(request2) {
      const maxLines = 20;
      const prefix = request2.text.slice(0, request2.position);
      const lines = splitLines(prefix);
      const cutoff = Math.max(lines.length - maxLines, 0);
      const prompt = lines.slice(cutoff).join("");
      return prompt;
    }
    setServerUrl(serverUrl) {
      this.serverUrl = serverUrl.replace(/\/$/, "");
      this.ping();
      this.api = new TabbyApi({ BASE: this.serverUrl });
      return this.serverUrl;
    }
    getServerUrl() {
      return this.serverUrl;
    }
    getStatus() {
      return this.status;
    }
    getCompletions(request2) {
      if (this.completionCache.has(request2)) {
        return new CancelablePromise((resolve2) => {
          resolve2(this.completionCache.get(request2));
        });
      }
      const prompt = this.createPrompt(request2);
      if (isBlank(prompt)) {
        return new CancelablePromise((resolve2) => {
          resolve2({
            id: "agent-" + v4_default(),
            created: (/* @__PURE__ */ new Date()).getTime(),
            choices: []
          });
        });
      }
      const promise = this.wrapApiPromise(this.api.default.completionsV1CompletionsPost({
        prompt,
        language: request2.language
      }));
      return cancelable(
        promise.then((response) => {
          this.completionCache.set(request2, response);
          return response;
        }),
        () => {
          promise.cancel();
        }
      );
    }
    postEvent(request2) {
      return this.wrapApiPromise(this.api.default.eventsV1EventsPost(request2));
    }
  };

  // src/types.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  var agentEventNames = ["statusChanged"];
  return __toCommonJS(src_exports);
})();
/*! Bundled license information:

@jspm/core/nodelibs/browser/buffer.js:
  (*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> *)

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
//# sourceMappingURL=index.global.js.map

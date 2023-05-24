var Tabby = (() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __esm = (fn, res) => function __init() {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  };
  var __commonJS = (cb, mod) => function __require() {
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
  function Item(fun, array) {
    this.fun = fun;
    this.array = array;
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
  var env, _performance, nowOffset, nanoPerSec;
  var init_process = __esm({
    "node_modules/@jspm/core/nodelibs/browser/process.js"() {
      init_global();
      init_dirname();
      init_filename();
      init_buffer2();
      init_process2();
      Item.prototype.run = function() {
        this.fun.apply(null, this.array);
      };
      env = {
        PATH: "/usr/bin",
        LANG: navigator.language + ".UTF-8",
        PWD: "/",
        HOME: "/home",
        TMP: "/tmp"
      };
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
    for (var i4 = 0, len = code.length; i4 < len; ++i4) {
      lookup[i4] = code[i4];
      revLookup[code.charCodeAt(i4)] = i4;
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
      var i5;
      for (i5 = 0; i5 < len2; i5 += 4) {
        tmp = revLookup[b64.charCodeAt(i5)] << 18 | revLookup[b64.charCodeAt(i5 + 1)] << 12 | revLookup[b64.charCodeAt(i5 + 2)] << 6 | revLookup[b64.charCodeAt(i5 + 3)];
        arr[curByte++] = tmp >> 16 & 255;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 2) {
        tmp = revLookup[b64.charCodeAt(i5)] << 2 | revLookup[b64.charCodeAt(i5 + 1)] >> 4;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 1) {
        tmp = revLookup[b64.charCodeAt(i5)] << 10 | revLookup[b64.charCodeAt(i5 + 1)] << 4 | revLookup[b64.charCodeAt(i5 + 2)] >> 2;
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
      for (var i5 = start; i5 < end; i5 += 3) {
        tmp = (uint8[i5] << 16 & 16711680) + (uint8[i5 + 1] << 8 & 65280) + (uint8[i5 + 2] & 255);
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
      for (var i5 = 0, len22 = len2 - extraBytes; i5 < len22; i5 += maxChunkLength) {
        parts.push(encodeChunk(uint8, i5, i5 + maxChunkLength > len22 ? len22 : i5 + maxChunkLength));
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
      var e5, m4;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var nBits = -7;
      var i4 = isLE ? nBytes - 1 : 0;
      var d4 = isLE ? -1 : 1;
      var s4 = buffer[offset + i4];
      i4 += d4;
      e5 = s4 & (1 << -nBits) - 1;
      s4 >>= -nBits;
      nBits += eLen;
      for (; nBits > 0; e5 = e5 * 256 + buffer[offset + i4], i4 += d4, nBits -= 8) {
      }
      m4 = e5 & (1 << -nBits) - 1;
      e5 >>= -nBits;
      nBits += mLen;
      for (; nBits > 0; m4 = m4 * 256 + buffer[offset + i4], i4 += d4, nBits -= 8) {
      }
      if (e5 === 0) {
        e5 = 1 - eBias;
      } else if (e5 === eMax) {
        return m4 ? NaN : (s4 ? -1 : 1) * Infinity;
      } else {
        m4 = m4 + Math.pow(2, mLen);
        e5 = e5 - eBias;
      }
      return (s4 ? -1 : 1) * m4 * Math.pow(2, e5 - mLen);
    };
    exports$2.write = function(buffer, value, offset, isLE, mLen, nBytes) {
      var e5, m4, c4;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
      var i4 = isLE ? 0 : nBytes - 1;
      var d4 = isLE ? 1 : -1;
      var s4 = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
      value = Math.abs(value);
      if (isNaN(value) || value === Infinity) {
        m4 = isNaN(value) ? 1 : 0;
        e5 = eMax;
      } else {
        e5 = Math.floor(Math.log(value) / Math.LN2);
        if (value * (c4 = Math.pow(2, -e5)) < 1) {
          e5--;
          c4 *= 2;
        }
        if (e5 + eBias >= 1) {
          value += rt / c4;
        } else {
          value += rt * Math.pow(2, 1 - eBias);
        }
        if (value * c4 >= 2) {
          e5++;
          c4 /= 2;
        }
        if (e5 + eBias >= eMax) {
          m4 = 0;
          e5 = eMax;
        } else if (e5 + eBias >= 1) {
          m4 = (value * c4 - 1) * Math.pow(2, mLen);
          e5 = e5 + eBias;
        } else {
          m4 = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
          e5 = 0;
        }
      }
      for (; mLen >= 8; buffer[offset + i4] = m4 & 255, i4 += d4, m4 /= 256, mLen -= 8) {
      }
      e5 = e5 << mLen | m4;
      eLen += mLen;
      for (; eLen > 0; buffer[offset + i4] = e5 & 255, i4 += d4, e5 /= 256, eLen -= 8) {
      }
      buffer[offset + i4 - d4] |= s4 * 128;
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
      } catch (e5) {
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
      const b3 = fromObject(value);
      if (b3)
        return b3;
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
      for (let i4 = 0; i4 < length; i4 += 1) {
        buf[i4] = array[i4] & 255;
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
    Buffer3.isBuffer = function isBuffer2(b3) {
      return b3 != null && b3._isBuffer === true && b3 !== Buffer3.prototype;
    };
    Buffer3.compare = function compare(a4, b3) {
      if (isInstance(a4, Uint8Array))
        a4 = Buffer3.from(a4, a4.offset, a4.byteLength);
      if (isInstance(b3, Uint8Array))
        b3 = Buffer3.from(b3, b3.offset, b3.byteLength);
      if (!Buffer3.isBuffer(a4) || !Buffer3.isBuffer(b3)) {
        throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');
      }
      if (a4 === b3)
        return 0;
      let x3 = a4.length;
      let y4 = b3.length;
      for (let i4 = 0, len = Math.min(x3, y4); i4 < len; ++i4) {
        if (a4[i4] !== b3[i4]) {
          x3 = a4[i4];
          y4 = b3[i4];
          break;
        }
      }
      if (x3 < y4)
        return -1;
      if (y4 < x3)
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
      let i4;
      if (length === void 0) {
        length = 0;
        for (i4 = 0; i4 < list.length; ++i4) {
          length += list[i4].length;
        }
      }
      const buffer = Buffer3.allocUnsafe(length);
      let pos = 0;
      for (i4 = 0; i4 < list.length; ++i4) {
        let buf = list[i4];
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
    function swap(b3, n4, m4) {
      const i4 = b3[n4];
      b3[n4] = b3[m4];
      b3[m4] = i4;
    }
    Buffer3.prototype.swap16 = function swap16() {
      const len = this.length;
      if (len % 2 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 16-bits");
      }
      for (let i4 = 0; i4 < len; i4 += 2) {
        swap(this, i4, i4 + 1);
      }
      return this;
    };
    Buffer3.prototype.swap32 = function swap32() {
      const len = this.length;
      if (len % 4 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 32-bits");
      }
      for (let i4 = 0; i4 < len; i4 += 4) {
        swap(this, i4, i4 + 3);
        swap(this, i4 + 1, i4 + 2);
      }
      return this;
    };
    Buffer3.prototype.swap64 = function swap64() {
      const len = this.length;
      if (len % 8 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 64-bits");
      }
      for (let i4 = 0; i4 < len; i4 += 8) {
        swap(this, i4, i4 + 7);
        swap(this, i4 + 1, i4 + 6);
        swap(this, i4 + 2, i4 + 5);
        swap(this, i4 + 3, i4 + 4);
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
    Buffer3.prototype.equals = function equals(b3) {
      if (!Buffer3.isBuffer(b3))
        throw new TypeError("Argument must be a Buffer");
      if (this === b3)
        return true;
      return Buffer3.compare(this, b3) === 0;
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
      let x3 = thisEnd - thisStart;
      let y4 = end - start;
      const len = Math.min(x3, y4);
      const thisCopy = this.slice(thisStart, thisEnd);
      const targetCopy = target.slice(start, end);
      for (let i4 = 0; i4 < len; ++i4) {
        if (thisCopy[i4] !== targetCopy[i4]) {
          x3 = thisCopy[i4];
          y4 = targetCopy[i4];
          break;
        }
      }
      if (x3 < y4)
        return -1;
      if (y4 < x3)
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
      function read(buf, i5) {
        if (indexSize === 1) {
          return buf[i5];
        } else {
          return buf.readUInt16BE(i5 * indexSize);
        }
      }
      let i4;
      if (dir) {
        let foundIndex = -1;
        for (i4 = byteOffset; i4 < arrLength; i4++) {
          if (read(arr, i4) === read(val, foundIndex === -1 ? 0 : i4 - foundIndex)) {
            if (foundIndex === -1)
              foundIndex = i4;
            if (i4 - foundIndex + 1 === valLength)
              return foundIndex * indexSize;
          } else {
            if (foundIndex !== -1)
              i4 -= i4 - foundIndex;
            foundIndex = -1;
          }
        }
      } else {
        if (byteOffset + valLength > arrLength)
          byteOffset = arrLength - valLength;
        for (i4 = byteOffset; i4 >= 0; i4--) {
          let found = true;
          for (let j3 = 0; j3 < valLength; j3++) {
            if (read(arr, i4 + j3) !== read(val, j3)) {
              found = false;
              break;
            }
          }
          if (found)
            return i4;
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
      let i4;
      for (i4 = 0; i4 < length; ++i4) {
        const parsed = parseInt(string.substr(i4 * 2, 2), 16);
        if (numberIsNaN(parsed))
          return i4;
        buf[offset + i4] = parsed;
      }
      return i4;
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
      let i4 = start;
      while (i4 < end) {
        const firstByte = buf[i4];
        let codePoint = null;
        let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
        if (i4 + bytesPerSequence <= end) {
          let secondByte, thirdByte, fourthByte, tempCodePoint;
          switch (bytesPerSequence) {
            case 1:
              if (firstByte < 128) {
                codePoint = firstByte;
              }
              break;
            case 2:
              secondByte = buf[i4 + 1];
              if ((secondByte & 192) === 128) {
                tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
                if (tempCodePoint > 127) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 3:
              secondByte = buf[i4 + 1];
              thirdByte = buf[i4 + 2];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
                if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 4:
              secondByte = buf[i4 + 1];
              thirdByte = buf[i4 + 2];
              fourthByte = buf[i4 + 3];
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
        i4 += bytesPerSequence;
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
      let i4 = 0;
      while (i4 < len) {
        res += String.fromCharCode.apply(String, codePoints.slice(i4, i4 += MAX_ARGUMENTS_LENGTH));
      }
      return res;
    }
    function asciiSlice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i4 = start; i4 < end; ++i4) {
        ret += String.fromCharCode(buf[i4] & 127);
      }
      return ret;
    }
    function latin1Slice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i4 = start; i4 < end; ++i4) {
        ret += String.fromCharCode(buf[i4]);
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
      for (let i4 = start; i4 < end; ++i4) {
        out += hexSliceLookupTable[buf[i4]];
      }
      return out;
    }
    function utf16leSlice(buf, start, end) {
      const bytes = buf.slice(start, end);
      let res = "";
      for (let i4 = 0; i4 < bytes.length - 1; i4 += 2) {
        res += String.fromCharCode(bytes[i4] + bytes[i4 + 1] * 256);
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
      let i4 = 0;
      while (++i4 < byteLength2 && (mul *= 256)) {
        val += this[offset + i4] * mul;
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
      let i4 = 0;
      while (++i4 < byteLength2 && (mul *= 256)) {
        val += this[offset + i4] * mul;
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
      let i4 = byteLength2;
      let mul = 1;
      let val = this[offset + --i4];
      while (i4 > 0 && (mul *= 256)) {
        val += this[offset + --i4] * mul;
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
      let i4 = 0;
      this[offset] = value & 255;
      while (++i4 < byteLength2 && (mul *= 256)) {
        this[offset + i4] = value / mul & 255;
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
      let i4 = byteLength2 - 1;
      let mul = 1;
      this[offset + i4] = value & 255;
      while (--i4 >= 0 && (mul *= 256)) {
        this[offset + i4] = value / mul & 255;
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
      let i4 = 0;
      let mul = 1;
      let sub = 0;
      this[offset] = value & 255;
      while (++i4 < byteLength2 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i4 - 1] !== 0) {
          sub = 1;
        }
        this[offset + i4] = (value / mul >> 0) - sub & 255;
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
      let i4 = byteLength2 - 1;
      let mul = 1;
      let sub = 0;
      this[offset + i4] = value & 255;
      while (--i4 >= 0 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i4 + 1] !== 0) {
          sub = 1;
        }
        this[offset + i4] = (value / mul >> 0) - sub & 255;
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
      let i4;
      if (typeof val === "number") {
        for (i4 = start; i4 < end; ++i4) {
          this[i4] = val;
        }
      } else {
        const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
        const len = bytes.length;
        if (len === 0) {
          throw new TypeError('The value "' + val + '" is invalid for argument "value"');
        }
        for (i4 = 0; i4 < end - start; ++i4) {
          this[i4 + start] = bytes[i4 % len];
        }
      }
      return this;
    };
    const errors = {};
    function E3(sym, getMessage, Base) {
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
    E3("ERR_BUFFER_OUT_OF_BOUNDS", function(name2) {
      if (name2) {
        return `${name2} is outside of buffer bounds`;
      }
      return "Attempt to access memory outside buffer bounds";
    }, RangeError);
    E3("ERR_INVALID_ARG_TYPE", function(name2, actual) {
      return `The "${name2}" argument must be of type number. Received type ${typeof actual}`;
    }, TypeError);
    E3("ERR_OUT_OF_RANGE", function(str, range, input) {
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
      let i4 = val.length;
      const start = val[0] === "-" ? 1 : 0;
      for (; i4 >= start + 4; i4 -= 3) {
        res = `_${val.slice(i4 - 3, i4)}${res}`;
      }
      return `${val.slice(0, i4)}${res}`;
    }
    function checkBounds(buf, offset, byteLength2) {
      validateNumber(offset, "offset");
      if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
        boundsError(offset, buf.length - (byteLength2 + 1));
      }
    }
    function checkIntBI(value, min, max, buf, offset, byteLength2) {
      if (value > max || value < min) {
        const n4 = typeof min === "bigint" ? "n" : "";
        let range;
        if (byteLength2 > 3) {
          if (min === 0 || min === BigInt(0)) {
            range = `>= 0${n4} and < 2${n4} ** ${(byteLength2 + 1) * 8}${n4}`;
          } else {
            range = `>= -(2${n4} ** ${(byteLength2 + 1) * 8 - 1}${n4}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n4}`;
          }
        } else {
          range = `>= ${min}${n4} and <= ${max}${n4}`;
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
      for (let i4 = 0; i4 < length; ++i4) {
        codePoint = string.charCodeAt(i4);
        if (codePoint > 55295 && codePoint < 57344) {
          if (!leadSurrogate) {
            if (codePoint > 56319) {
              if ((units -= 3) > -1)
                bytes.push(239, 191, 189);
              continue;
            } else if (i4 + 1 === length) {
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
      for (let i4 = 0; i4 < str.length; ++i4) {
        byteArray.push(str.charCodeAt(i4) & 255);
      }
      return byteArray;
    }
    function utf16leToBytes(str, units) {
      let c4, hi, lo;
      const byteArray = [];
      for (let i4 = 0; i4 < str.length; ++i4) {
        if ((units -= 2) < 0)
          break;
        c4 = str.charCodeAt(i4);
        hi = c4 >> 8;
        lo = c4 % 256;
        byteArray.push(lo);
        byteArray.push(hi);
      }
      return byteArray;
    }
    function base64ToBytes(str) {
      return base642.toByteArray(base64clean(str));
    }
    function blitBuffer(src, dst, offset, length) {
      let i4;
      for (i4 = 0; i4 < length; ++i4) {
        if (i4 + offset >= dst.length || i4 >= src.length)
          break;
        dst[i4 + offset] = src[i4];
      }
      return i4;
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
      for (let i4 = 0; i4 < 16; ++i4) {
        const i16 = i4 * 16;
        for (let j3 = 0; j3 < 16; ++j3) {
          table[i16 + j3] = alphabet[i4] + alphabet[j3];
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
    let i4;
    let l4;
    if (typeof obj !== "object") {
      obj = [obj];
    }
    if (isArray(obj)) {
      for (i4 = 0, l4 = obj.length; i4 < l4; i4++) {
        fn.call(null, obj[i4], i4, obj);
      }
    } else {
      const keys = allOwnKeys ? Object.getOwnPropertyNames(obj) : Object.keys(obj);
      const len = keys.length;
      let key;
      for (i4 = 0; i4 < len; i4++) {
        key = keys[i4];
        fn.call(null, obj[key], key, obj);
      }
    }
  }
  function findKey(obj, key) {
    key = key.toLowerCase();
    const keys = Object.keys(obj);
    let i4 = keys.length;
    let _key;
    while (i4-- > 0) {
      _key = keys[i4];
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
    for (let i4 = 0, l4 = arguments.length; i4 < l4; i4++) {
      arguments[i4] && forEach(arguments[i4], assignValue);
    }
    return result;
  }
  var extend = (a4, b3, thisArg, { allOwnKeys } = {}) => {
    forEach(b3, (val, key) => {
      if (thisArg && isFunction(val)) {
        a4[key] = bind(val, thisArg);
      } else {
        a4[key] = val;
      }
    }, { allOwnKeys });
    return a4;
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
    let i4;
    let prop;
    const merged = {};
    destObj = destObj || {};
    if (sourceObj == null)
      return destObj;
    do {
      props = Object.getOwnPropertyNames(sourceObj);
      i4 = props.length;
      while (i4-- > 0) {
        prop = props[i4];
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
    let i4 = thing.length;
    if (!isNumber(i4))
      return null;
    const arr = new Array(i4);
    while (i4-- > 0) {
      arr[i4] = thing[i4];
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
      function replacer(m4, p1, p22) {
        return p1.toUpperCase() + p22;
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
    const define = (arr) => {
      arr.forEach((value) => {
        obj[value] = true;
      });
    };
    isArray(arrayOrString) ? define(arrayOrString) : define(String(arrayOrString).split(delimiter));
    return obj;
  };
  var noop = () => {
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
    const visit = (source, i4) => {
      if (isObject(source)) {
        if (stack.indexOf(source) >= 0) {
          return;
        }
        if (!("toJSON" in source)) {
          stack[i4] = source;
          const target = isArray(source) ? [] : {};
          forEach(source, (value, key) => {
            const reducedValue = visit(value, i4 + 1);
            !isUndefined(reducedValue) && (target[key] = reducedValue);
          });
          stack[i4] = void 0;
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
    noop,
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
  function AxiosError(message, code, config, request2, response) {
    Error.call(this);
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    } else {
      this.stack = new Error().stack;
    }
    this.message = message;
    this.name = "AxiosError";
    code && (this.code = code);
    config && (this.config = config);
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
  AxiosError.from = (error, code, config, request2, response, customProps) => {
    const axiosError = Object.create(prototype);
    utils_default.toFlatObject(error, axiosError, function filter2(obj) {
      return obj !== Error.prototype;
    }, (prop) => {
      return prop !== "isAxiosError";
    });
    AxiosError.call(axiosError, error.message, code, config, request2, response);
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
    return path.concat(key).map(function each(token, i4) {
      token = removeBrackets(token);
      return !dots && i4 ? "[" + token + "]" : token;
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
      utils_default.forEach(this.handlers, function forEachHandler(h5) {
        if (h5 !== null) {
          fn(h5);
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
    let i4;
    const len = keys.length;
    let key;
    for (i4 = 0; i4 < len; i4++) {
      key = keys[i4];
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
      } catch (e5) {
        if (e5.name !== "SyntaxError") {
          throw e5;
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
        } catch (e5) {
          if (strictJSONParsing) {
            if (e5.name === "SyntaxError") {
              throw AxiosError_default.from(e5, AxiosError_default.ERR_BAD_RESPONSE, this, null, this.response);
            }
            throw e5;
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
    let i4;
    rawHeaders && rawHeaders.split("\n").forEach(function parser(line) {
      i4 = line.indexOf(":");
      key = line.substring(0, i4).trim().toLowerCase();
      val = line.substring(i4 + 1).trim();
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
    return header.trim().toLowerCase().replace(/([a-z\d])(\w*)/g, (w3, char, str) => {
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
      let i4 = keys.length;
      let deleted = false;
      while (i4--) {
        const key = keys[i4];
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
    const config = this || defaults_default;
    const context = response || config;
    const headers = AxiosHeaders_default.from(context.headers);
    let data = context.data;
    utils_default.forEach(fns, function transform(fn) {
      data = fn.call(config, data, headers.normalize(), response ? response.status : void 0);
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
  function CanceledError(message, config, request2) {
    AxiosError_default.call(this, message == null ? "canceled" : message, AxiosError_default.ERR_CANCELED, config, request2);
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
        write: function write(name2, value, expires, path, domain, secure) {
          const cookie = [];
          cookie.push(name2 + "=" + encodeURIComponent(value));
          if (utils_default.isNumber(expires)) {
            cookie.push("expires=" + new Date(expires).toGMTString());
          }
          if (utils_default.isString(path)) {
            cookie.push("path=" + path);
          }
          if (utils_default.isString(domain)) {
            cookie.push("domain=" + domain);
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
      let i4 = tail;
      let bytesCount = 0;
      while (i4 !== head) {
        bytesCount += bytes[i4++];
        i4 = i4 % samplesCount;
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
    return (e5) => {
      const loaded = e5.loaded;
      const total = e5.lengthComputable ? e5.total : void 0;
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
        event: e5
      };
      data[isDownloadStream ? "download" : "upload"] = true;
      listener(data);
    };
  }
  var isXHRAdapterSupported = typeof XMLHttpRequest !== "undefined";
  var xhr_default = isXHRAdapterSupported && function(config) {
    return new Promise(function dispatchXhrRequest(resolve2, reject) {
      let requestData = config.data;
      const requestHeaders = AxiosHeaders_default.from(config.headers).normalize();
      const responseType = config.responseType;
      let onCanceled;
      function done() {
        if (config.cancelToken) {
          config.cancelToken.unsubscribe(onCanceled);
        }
        if (config.signal) {
          config.signal.removeEventListener("abort", onCanceled);
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
      if (config.auth) {
        const username = config.auth.username || "";
        const password = config.auth.password ? unescape(encodeURIComponent(config.auth.password)) : "";
        requestHeaders.set("Authorization", "Basic " + btoa(username + ":" + password));
      }
      const fullPath = buildFullPath(config.baseURL, config.url);
      request2.open(config.method.toUpperCase(), buildURL(fullPath, config.params, config.paramsSerializer), true);
      request2.timeout = config.timeout;
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
          config,
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
        reject(new AxiosError_default("Request aborted", AxiosError_default.ECONNABORTED, config, request2));
        request2 = null;
      };
      request2.onerror = function handleError() {
        reject(new AxiosError_default("Network Error", AxiosError_default.ERR_NETWORK, config, request2));
        request2 = null;
      };
      request2.ontimeout = function handleTimeout() {
        let timeoutErrorMessage = config.timeout ? "timeout of " + config.timeout + "ms exceeded" : "timeout exceeded";
        const transitional2 = config.transitional || transitional_default;
        if (config.timeoutErrorMessage) {
          timeoutErrorMessage = config.timeoutErrorMessage;
        }
        reject(new AxiosError_default(
          timeoutErrorMessage,
          transitional2.clarifyTimeoutError ? AxiosError_default.ETIMEDOUT : AxiosError_default.ECONNABORTED,
          config,
          request2
        ));
        request2 = null;
      };
      if (browser_default.isStandardBrowserEnv) {
        const xsrfValue = (config.withCredentials || isURLSameOrigin_default(fullPath)) && config.xsrfCookieName && cookies_default.read(config.xsrfCookieName);
        if (xsrfValue) {
          requestHeaders.set(config.xsrfHeaderName, xsrfValue);
        }
      }
      requestData === void 0 && requestHeaders.setContentType(null);
      if ("setRequestHeader" in request2) {
        utils_default.forEach(requestHeaders.toJSON(), function setRequestHeader(val, key) {
          request2.setRequestHeader(key, val);
        });
      }
      if (!utils_default.isUndefined(config.withCredentials)) {
        request2.withCredentials = !!config.withCredentials;
      }
      if (responseType && responseType !== "json") {
        request2.responseType = config.responseType;
      }
      if (typeof config.onDownloadProgress === "function") {
        request2.addEventListener("progress", progressEventReducer(config.onDownloadProgress, true));
      }
      if (typeof config.onUploadProgress === "function" && request2.upload) {
        request2.upload.addEventListener("progress", progressEventReducer(config.onUploadProgress));
      }
      if (config.cancelToken || config.signal) {
        onCanceled = (cancel) => {
          if (!request2) {
            return;
          }
          reject(!cancel || cancel.type ? new CanceledError_default(null, config, request2) : cancel);
          request2.abort();
          request2 = null;
        };
        config.cancelToken && config.cancelToken.subscribe(onCanceled);
        if (config.signal) {
          config.signal.aborted ? onCanceled() : config.signal.addEventListener("abort", onCanceled);
        }
      }
      const protocol = parseProtocol(fullPath);
      if (protocol && browser_default.protocols.indexOf(protocol) === -1) {
        reject(new AxiosError_default("Unsupported protocol " + protocol + ":", AxiosError_default.ERR_BAD_REQUEST, config));
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
      } catch (e5) {
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
      for (let i4 = 0; i4 < length; i4++) {
        nameOrAdapter = adapters[i4];
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
  function throwIfCancellationRequested(config) {
    if (config.cancelToken) {
      config.cancelToken.throwIfRequested();
    }
    if (config.signal && config.signal.aborted) {
      throw new CanceledError_default(null, config);
    }
  }
  function dispatchRequest(config) {
    throwIfCancellationRequested(config);
    config.headers = AxiosHeaders_default.from(config.headers);
    config.data = transformData.call(
      config,
      config.transformRequest
    );
    if (["post", "put", "patch"].indexOf(config.method) !== -1) {
      config.headers.setContentType("application/x-www-form-urlencoded", false);
    }
    const adapter = adapters_default.getAdapter(config.adapter || defaults_default.adapter);
    return adapter(config).then(function onAdapterResolution(response) {
      throwIfCancellationRequested(config);
      response.data = transformData.call(
        config,
        config.transformResponse,
        response
      );
      response.headers = AxiosHeaders_default.from(response.headers);
      return response;
    }, function onAdapterRejection(reason) {
      if (!isCancel(reason)) {
        throwIfCancellationRequested(config);
        if (reason && reason.response) {
          reason.response.data = transformData.call(
            config,
            config.transformResponse,
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
    const config = {};
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
    function mergeDeepProperties(a4, b3, caseless) {
      if (!utils_default.isUndefined(b3)) {
        return getMergedValue(a4, b3, caseless);
      } else if (!utils_default.isUndefined(a4)) {
        return getMergedValue(void 0, a4, caseless);
      }
    }
    function valueFromConfig2(a4, b3) {
      if (!utils_default.isUndefined(b3)) {
        return getMergedValue(void 0, b3);
      }
    }
    function defaultToConfig2(a4, b3) {
      if (!utils_default.isUndefined(b3)) {
        return getMergedValue(void 0, b3);
      } else if (!utils_default.isUndefined(a4)) {
        return getMergedValue(void 0, a4);
      }
    }
    function mergeDirectKeys(a4, b3, prop) {
      if (prop in config2) {
        return getMergedValue(a4, b3);
      } else if (prop in config1) {
        return getMergedValue(void 0, a4);
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
      headers: (a4, b3) => mergeDeepProperties(headersToObject(a4), headersToObject(b3), true)
    };
    utils_default.forEach(Object.keys(Object.assign({}, config1, config2)), function computeConfigValue(prop) {
      const merge2 = mergeMap[prop] || mergeDeepProperties;
      const configValue = merge2(config1[prop], config2[prop], prop);
      utils_default.isUndefined(configValue) && merge2 !== mergeDirectKeys || (config[prop] = configValue);
    });
    return config;
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
  ["object", "boolean", "number", "function", "string", "symbol"].forEach((type, i4) => {
    validators[type] = function validator(thing) {
      return typeof thing === type || "a" + (i4 < 1 ? "n " : " ") + type;
    };
  });
  var deprecatedWarnings = {};
  validators.transitional = function transitional(validator, version, message) {
    function formatMessage(opt, desc) {
      return "[Axios v" + VERSION + "] Transitional option '" + opt + "'" + desc + (message ? ". " + message : "");
    }
    return (value, opt, opts) => {
      if (validator === false) {
        throw new AxiosError_default(
          formatMessage(opt, " has been removed" + (version ? " in " + version : "")),
          AxiosError_default.ERR_DEPRECATED
        );
      }
      if (version && !deprecatedWarnings[opt]) {
        deprecatedWarnings[opt] = true;
        console.warn(
          formatMessage(
            opt,
            " has been deprecated since v" + version + " and will be removed in the near future"
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
    let i4 = keys.length;
    while (i4-- > 0) {
      const opt = keys[i4];
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
    request(configOrUrl, config) {
      if (typeof configOrUrl === "string") {
        config = config || {};
        config.url = configOrUrl;
      } else {
        config = configOrUrl || {};
      }
      config = mergeConfig(this.defaults, config);
      const { transitional: transitional2, paramsSerializer, headers } = config;
      if (transitional2 !== void 0) {
        validator_default.assertOptions(transitional2, {
          silentJSONParsing: validators2.transitional(validators2.boolean),
          forcedJSONParsing: validators2.transitional(validators2.boolean),
          clarifyTimeoutError: validators2.transitional(validators2.boolean)
        }, false);
      }
      if (paramsSerializer != null) {
        if (utils_default.isFunction(paramsSerializer)) {
          config.paramsSerializer = {
            serialize: paramsSerializer
          };
        } else {
          validator_default.assertOptions(paramsSerializer, {
            encode: validators2.function,
            serialize: validators2.function
          }, true);
        }
      }
      config.method = (config.method || this.defaults.method || "get").toLowerCase();
      let contextHeaders;
      contextHeaders = headers && utils_default.merge(
        headers.common,
        headers[config.method]
      );
      contextHeaders && utils_default.forEach(
        ["delete", "get", "head", "post", "put", "patch", "common"],
        (method) => {
          delete headers[method];
        }
      );
      config.headers = AxiosHeaders_default.concat(contextHeaders, headers);
      const requestInterceptorChain = [];
      let synchronousRequestInterceptors = true;
      this.interceptors.request.forEach(function unshiftRequestInterceptors(interceptor) {
        if (typeof interceptor.runWhen === "function" && interceptor.runWhen(config) === false) {
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
      let i4 = 0;
      let len;
      if (!synchronousRequestInterceptors) {
        const chain = [dispatchRequest.bind(this), void 0];
        chain.unshift.apply(chain, requestInterceptorChain);
        chain.push.apply(chain, responseInterceptorChain);
        len = chain.length;
        promise = Promise.resolve(config);
        while (i4 < len) {
          promise = promise.then(chain[i4++], chain[i4++]);
        }
        return promise;
      }
      len = requestInterceptorChain.length;
      let newConfig = config;
      i4 = 0;
      while (i4 < len) {
        const onFulfilled = requestInterceptorChain[i4++];
        const onRejected = requestInterceptorChain[i4++];
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
      i4 = 0;
      len = responseInterceptorChain.length;
      while (i4 < len) {
        promise = promise.then(responseInterceptorChain[i4++], responseInterceptorChain[i4++]);
      }
      return promise;
    }
    getUri(config) {
      config = mergeConfig(this.defaults, config);
      const fullPath = buildFullPath(config.baseURL, config.url);
      return buildURL(fullPath, config.params, config.paramsSerializer);
    }
  };
  utils_default.forEach(["delete", "get", "head", "options"], function forEachMethodNoData2(method) {
    Axios.prototype[method] = function(url, config) {
      return this.request(mergeConfig(config || {}, {
        method,
        url,
        data: (config || {}).data
      }));
    };
  });
  utils_default.forEach(["post", "put", "patch"], function forEachMethodWithData2(method) {
    function generateHTTPMethod(isForm) {
      return function httpMethod(url, data, config) {
        return this.request(mergeConfig(config || {}, {
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
        let i4 = token._listeners.length;
        while (i4-- > 0) {
          token._listeners[i4](cancel);
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
      executor(function cancel(message, config, request2) {
        if (token.reason) {
          return;
        }
        token.reason = new CanceledError_default(message, config, request2);
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
      const token = new CancelToken(function executor(c4) {
        cancel = c4;
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
  var r = n && "function" == typeof n.apply ? n.apply : function(e5, t5, n4) {
    return Function.prototype.apply.call(e5, t5, n4);
  };
  t = n && "function" == typeof n.ownKeys ? n.ownKeys : Object.getOwnPropertySymbols ? function(e5) {
    return Object.getOwnPropertyNames(e5).concat(Object.getOwnPropertySymbols(e5));
  } : function(e5) {
    return Object.getOwnPropertyNames(e5);
  };
  var i = Number.isNaN || function(e5) {
    return e5 != e5;
  };
  function o() {
    o.init.call(this);
  }
  e = o, o.EventEmitter = o, o.prototype._events = void 0, o.prototype._eventsCount = 0, o.prototype._maxListeners = void 0;
  var s = 10;
  function u(e5) {
    if ("function" != typeof e5)
      throw new TypeError('The "listener" argument must be of type Function. Received type ' + typeof e5);
  }
  function f(e5) {
    return void 0 === e5._maxListeners ? o.defaultMaxListeners : e5._maxListeners;
  }
  function v(e5, t5, n4, r5) {
    var i4, o4, s4, v4;
    if (u(n4), void 0 === (o4 = e5._events) ? (o4 = e5._events = /* @__PURE__ */ Object.create(null), e5._eventsCount = 0) : (void 0 !== o4.newListener && (e5.emit("newListener", t5, n4.listener ? n4.listener : n4), o4 = e5._events), s4 = o4[t5]), void 0 === s4)
      s4 = o4[t5] = n4, ++e5._eventsCount;
    else if ("function" == typeof s4 ? s4 = o4[t5] = r5 ? [n4, s4] : [s4, n4] : r5 ? s4.unshift(n4) : s4.push(n4), (i4 = f(e5)) > 0 && s4.length > i4 && !s4.warned) {
      s4.warned = true;
      var a4 = new Error("Possible EventEmitter memory leak detected. " + s4.length + " " + String(t5) + " listeners added. Use emitter.setMaxListeners() to increase limit");
      a4.name = "MaxListenersExceededWarning", a4.emitter = e5, a4.type = t5, a4.count = s4.length, v4 = a4, console && console.warn && console.warn(v4);
    }
    return e5;
  }
  function a() {
    if (!this.fired)
      return this.target.removeListener(this.type, this.wrapFn), this.fired = true, 0 === arguments.length ? this.listener.call(this.target) : this.listener.apply(this.target, arguments);
  }
  function l(e5, t5, n4) {
    var r5 = { fired: false, wrapFn: void 0, target: e5, type: t5, listener: n4 }, i4 = a.bind(r5);
    return i4.listener = n4, r5.wrapFn = i4, i4;
  }
  function h(e5, t5, n4) {
    var r5 = e5._events;
    if (void 0 === r5)
      return [];
    var i4 = r5[t5];
    return void 0 === i4 ? [] : "function" == typeof i4 ? n4 ? [i4.listener || i4] : [i4] : n4 ? function(e6) {
      for (var t6 = new Array(e6.length), n5 = 0; n5 < t6.length; ++n5)
        t6[n5] = e6[n5].listener || e6[n5];
      return t6;
    }(i4) : c(i4, i4.length);
  }
  function p(e5) {
    var t5 = this._events;
    if (void 0 !== t5) {
      var n4 = t5[e5];
      if ("function" == typeof n4)
        return 1;
      if (void 0 !== n4)
        return n4.length;
    }
    return 0;
  }
  function c(e5, t5) {
    for (var n4 = new Array(t5), r5 = 0; r5 < t5; ++r5)
      n4[r5] = e5[r5];
    return n4;
  }
  Object.defineProperty(o, "defaultMaxListeners", { enumerable: true, get: function() {
    return s;
  }, set: function(e5) {
    if ("number" != typeof e5 || e5 < 0 || i(e5))
      throw new RangeError('The value of "defaultMaxListeners" is out of range. It must be a non-negative number. Received ' + e5 + ".");
    s = e5;
  } }), o.init = function() {
    void 0 !== this._events && this._events !== Object.getPrototypeOf(this)._events || (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0), this._maxListeners = this._maxListeners || void 0;
  }, o.prototype.setMaxListeners = function(e5) {
    if ("number" != typeof e5 || e5 < 0 || i(e5))
      throw new RangeError('The value of "n" is out of range. It must be a non-negative number. Received ' + e5 + ".");
    return this._maxListeners = e5, this;
  }, o.prototype.getMaxListeners = function() {
    return f(this);
  }, o.prototype.emit = function(e5) {
    for (var t5 = [], n4 = 1; n4 < arguments.length; n4++)
      t5.push(arguments[n4]);
    var i4 = "error" === e5, o4 = this._events;
    if (void 0 !== o4)
      i4 = i4 && void 0 === o4.error;
    else if (!i4)
      return false;
    if (i4) {
      var s4;
      if (t5.length > 0 && (s4 = t5[0]), s4 instanceof Error)
        throw s4;
      var u4 = new Error("Unhandled error." + (s4 ? " (" + s4.message + ")" : ""));
      throw u4.context = s4, u4;
    }
    var f4 = o4[e5];
    if (void 0 === f4)
      return false;
    if ("function" == typeof f4)
      r(f4, this, t5);
    else {
      var v4 = f4.length, a4 = c(f4, v4);
      for (n4 = 0; n4 < v4; ++n4)
        r(a4[n4], this, t5);
    }
    return true;
  }, o.prototype.addListener = function(e5, t5) {
    return v(this, e5, t5, false);
  }, o.prototype.on = o.prototype.addListener, o.prototype.prependListener = function(e5, t5) {
    return v(this, e5, t5, true);
  }, o.prototype.once = function(e5, t5) {
    return u(t5), this.on(e5, l(this, e5, t5)), this;
  }, o.prototype.prependOnceListener = function(e5, t5) {
    return u(t5), this.prependListener(e5, l(this, e5, t5)), this;
  }, o.prototype.removeListener = function(e5, t5) {
    var n4, r5, i4, o4, s4;
    if (u(t5), void 0 === (r5 = this._events))
      return this;
    if (void 0 === (n4 = r5[e5]))
      return this;
    if (n4 === t5 || n4.listener === t5)
      0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : (delete r5[e5], r5.removeListener && this.emit("removeListener", e5, n4.listener || t5));
    else if ("function" != typeof n4) {
      for (i4 = -1, o4 = n4.length - 1; o4 >= 0; o4--)
        if (n4[o4] === t5 || n4[o4].listener === t5) {
          s4 = n4[o4].listener, i4 = o4;
          break;
        }
      if (i4 < 0)
        return this;
      0 === i4 ? n4.shift() : !function(e6, t6) {
        for (; t6 + 1 < e6.length; t6++)
          e6[t6] = e6[t6 + 1];
        e6.pop();
      }(n4, i4), 1 === n4.length && (r5[e5] = n4[0]), void 0 !== r5.removeListener && this.emit("removeListener", e5, s4 || t5);
    }
    return this;
  }, o.prototype.off = o.prototype.removeListener, o.prototype.removeAllListeners = function(e5) {
    var t5, n4, r5;
    if (void 0 === (n4 = this._events))
      return this;
    if (void 0 === n4.removeListener)
      return 0 === arguments.length ? (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0) : void 0 !== n4[e5] && (0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : delete n4[e5]), this;
    if (0 === arguments.length) {
      var i4, o4 = Object.keys(n4);
      for (r5 = 0; r5 < o4.length; ++r5)
        "removeListener" !== (i4 = o4[r5]) && this.removeAllListeners(i4);
      return this.removeAllListeners("removeListener"), this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0, this;
    }
    if ("function" == typeof (t5 = n4[e5]))
      this.removeListener(e5, t5);
    else if (void 0 !== t5)
      for (r5 = t5.length - 1; r5 >= 0; r5--)
        this.removeListener(e5, t5[r5]);
    return this;
  }, o.prototype.listeners = function(e5) {
    return h(this, e5, true);
  }, o.prototype.rawListeners = function(e5) {
    return h(this, e5, false);
  }, o.listenerCount = function(e5, t5) {
    return "function" == typeof e5.listenerCount ? e5.listenerCount(t5) : p.call(e5, t5);
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
          const p4 = Promise.reject(error);
          error = null;
          return p4;
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
    on,
    once
  } = y;

  // node_modules/@jspm/core/nodelibs/browser/assert.js
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();

  // node_modules/@jspm/core/nodelibs/browser/chunk-b4205b57.js
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();

  // node_modules/@jspm/core/nodelibs/browser/chunk-5decc758.js
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  var e2;
  var t2;
  var n2;
  var r2 = "undefined" != typeof globalThis ? globalThis : "undefined" != typeof self ? self : global;
  var o2 = e2 = {};
  function i2() {
    throw new Error("setTimeout has not been defined");
  }
  function u2() {
    throw new Error("clearTimeout has not been defined");
  }
  function c2(e5) {
    if (t2 === setTimeout)
      return setTimeout(e5, 0);
    if ((t2 === i2 || !t2) && setTimeout)
      return t2 = setTimeout, setTimeout(e5, 0);
    try {
      return t2(e5, 0);
    } catch (n4) {
      try {
        return t2.call(null, e5, 0);
      } catch (n5) {
        return t2.call(this || r2, e5, 0);
      }
    }
  }
  !function() {
    try {
      t2 = "function" == typeof setTimeout ? setTimeout : i2;
    } catch (e5) {
      t2 = i2;
    }
    try {
      n2 = "function" == typeof clearTimeout ? clearTimeout : u2;
    } catch (e5) {
      n2 = u2;
    }
  }();
  var l2;
  var s2 = [];
  var f2 = false;
  var a2 = -1;
  function h2() {
    f2 && l2 && (f2 = false, l2.length ? s2 = l2.concat(s2) : a2 = -1, s2.length && d());
  }
  function d() {
    if (!f2) {
      var e5 = c2(h2);
      f2 = true;
      for (var t5 = s2.length; t5; ) {
        for (l2 = s2, s2 = []; ++a2 < t5; )
          l2 && l2[a2].run();
        a2 = -1, t5 = s2.length;
      }
      l2 = null, f2 = false, function(e6) {
        if (n2 === clearTimeout)
          return clearTimeout(e6);
        if ((n2 === u2 || !n2) && clearTimeout)
          return n2 = clearTimeout, clearTimeout(e6);
        try {
          n2(e6);
        } catch (t6) {
          try {
            return n2.call(null, e6);
          } catch (t7) {
            return n2.call(this || r2, e6);
          }
        }
      }(e5);
    }
  }
  function m(e5, t5) {
    (this || r2).fun = e5, (this || r2).array = t5;
  }
  function p2() {
  }
  o2.nextTick = function(e5) {
    var t5 = new Array(arguments.length - 1);
    if (arguments.length > 1)
      for (var n4 = 1; n4 < arguments.length; n4++)
        t5[n4 - 1] = arguments[n4];
    s2.push(new m(e5, t5)), 1 !== s2.length || f2 || c2(d);
  }, m.prototype.run = function() {
    (this || r2).fun.apply(null, (this || r2).array);
  }, o2.title = "browser", o2.browser = true, o2.env = {}, o2.argv = [], o2.version = "", o2.versions = {}, o2.on = p2, o2.addListener = p2, o2.once = p2, o2.off = p2, o2.removeListener = p2, o2.removeAllListeners = p2, o2.emit = p2, o2.prependListener = p2, o2.prependOnceListener = p2, o2.listeners = function(e5) {
    return [];
  }, o2.binding = function(e5) {
    throw new Error("process.binding is not supported");
  }, o2.cwd = function() {
    return "/";
  }, o2.chdir = function(e5) {
    throw new Error("process.chdir is not supported");
  }, o2.umask = function() {
    return 0;
  };
  var T = e2;
  T.addListener;
  T.argv;
  T.binding;
  T.browser;
  T.chdir;
  T.cwd;
  T.emit;
  T.env;
  T.listeners;
  T.nextTick;
  T.off;
  T.on;
  T.once;
  T.prependListener;
  T.prependOnceListener;
  T.removeAllListeners;
  T.removeListener;
  T.title;
  T.umask;
  T.version;
  T.versions;

  // node_modules/@jspm/core/nodelibs/browser/chunk-b4205b57.js
  var t3 = "function" == typeof Symbol && "symbol" == typeof Symbol.toStringTag;
  var e3 = Object.prototype.toString;
  var o3 = function(o4) {
    return !(t3 && o4 && "object" == typeof o4 && Symbol.toStringTag in o4) && "[object Arguments]" === e3.call(o4);
  };
  var n3 = function(t5) {
    return !!o3(t5) || null !== t5 && "object" == typeof t5 && "number" == typeof t5.length && t5.length >= 0 && "[object Array]" !== e3.call(t5) && "[object Function]" === e3.call(t5.callee);
  };
  var r3 = function() {
    return o3(arguments);
  }();
  o3.isLegacyArguments = n3;
  var l3 = r3 ? o3 : n3;
  var t$1 = Object.prototype.toString;
  var o$1 = Function.prototype.toString;
  var n$1 = /^\s*(?:function)?\*/;
  var e$1 = "function" == typeof Symbol && "symbol" == typeof Symbol.toStringTag;
  var r$1 = Object.getPrototypeOf;
  var c3 = function() {
    if (!e$1)
      return false;
    try {
      return Function("return function*() {}")();
    } catch (t5) {
    }
  }();
  var u3 = c3 ? r$1(c3) : {};
  var i3 = function(c4) {
    return "function" == typeof c4 && (!!n$1.test(o$1.call(c4)) || (e$1 ? r$1(c4) === u3 : "[object GeneratorFunction]" === t$1.call(c4)));
  };
  var t$2 = "function" == typeof Object.create ? function(t5, e5) {
    e5 && (t5.super_ = e5, t5.prototype = Object.create(e5.prototype, { constructor: { value: t5, enumerable: false, writable: true, configurable: true } }));
  } : function(t5, e5) {
    if (e5) {
      t5.super_ = e5;
      var o4 = function() {
      };
      o4.prototype = e5.prototype, t5.prototype = new o4(), t5.prototype.constructor = t5;
    }
  };
  var i$1 = function(e5) {
    return e5 && "object" == typeof e5 && "function" == typeof e5.copy && "function" == typeof e5.fill && "function" == typeof e5.readUInt8;
  };
  var o$2 = {};
  var u$1 = i$1;
  var f3 = l3;
  var a3 = i3;
  function c$1(e5) {
    return e5.call.bind(e5);
  }
  var s3 = "undefined" != typeof BigInt;
  var p3 = "undefined" != typeof Symbol;
  var y2 = p3 && void 0 !== Symbol.toStringTag;
  var l$1 = "undefined" != typeof Uint8Array;
  var d2 = "undefined" != typeof ArrayBuffer;
  if (l$1 && y2)
    var g = Object.getPrototypeOf(Uint8Array.prototype), b = c$1(Object.getOwnPropertyDescriptor(g, Symbol.toStringTag).get);
  var m2 = c$1(Object.prototype.toString);
  var h3 = c$1(Number.prototype.valueOf);
  var j = c$1(String.prototype.valueOf);
  var A = c$1(Boolean.prototype.valueOf);
  if (s3)
    var w = c$1(BigInt.prototype.valueOf);
  if (p3)
    var v2 = c$1(Symbol.prototype.valueOf);
  function O(e5, t5) {
    if ("object" != typeof e5)
      return false;
    try {
      return t5(e5), true;
    } catch (e6) {
      return false;
    }
  }
  function S(e5) {
    return l$1 && y2 ? void 0 !== b(e5) : B(e5) || k(e5) || E(e5) || D(e5) || U(e5) || P(e5) || x(e5) || I(e5) || M(e5) || z(e5) || F(e5);
  }
  function B(e5) {
    return l$1 && y2 ? "Uint8Array" === b(e5) : "[object Uint8Array]" === m2(e5) || u$1(e5) && void 0 !== e5.buffer;
  }
  function k(e5) {
    return l$1 && y2 ? "Uint8ClampedArray" === b(e5) : "[object Uint8ClampedArray]" === m2(e5);
  }
  function E(e5) {
    return l$1 && y2 ? "Uint16Array" === b(e5) : "[object Uint16Array]" === m2(e5);
  }
  function D(e5) {
    return l$1 && y2 ? "Uint32Array" === b(e5) : "[object Uint32Array]" === m2(e5);
  }
  function U(e5) {
    return l$1 && y2 ? "Int8Array" === b(e5) : "[object Int8Array]" === m2(e5);
  }
  function P(e5) {
    return l$1 && y2 ? "Int16Array" === b(e5) : "[object Int16Array]" === m2(e5);
  }
  function x(e5) {
    return l$1 && y2 ? "Int32Array" === b(e5) : "[object Int32Array]" === m2(e5);
  }
  function I(e5) {
    return l$1 && y2 ? "Float32Array" === b(e5) : "[object Float32Array]" === m2(e5);
  }
  function M(e5) {
    return l$1 && y2 ? "Float64Array" === b(e5) : "[object Float64Array]" === m2(e5);
  }
  function z(e5) {
    return l$1 && y2 ? "BigInt64Array" === b(e5) : "[object BigInt64Array]" === m2(e5);
  }
  function F(e5) {
    return l$1 && y2 ? "BigUint64Array" === b(e5) : "[object BigUint64Array]" === m2(e5);
  }
  function T2(e5) {
    return "[object Map]" === m2(e5);
  }
  function N(e5) {
    return "[object Set]" === m2(e5);
  }
  function W(e5) {
    return "[object WeakMap]" === m2(e5);
  }
  function $(e5) {
    return "[object WeakSet]" === m2(e5);
  }
  function C(e5) {
    return "[object ArrayBuffer]" === m2(e5);
  }
  function V(e5) {
    return "undefined" != typeof ArrayBuffer && (C.working ? C(e5) : e5 instanceof ArrayBuffer);
  }
  function G(e5) {
    return "[object DataView]" === m2(e5);
  }
  function R(e5) {
    return "undefined" != typeof DataView && (G.working ? G(e5) : e5 instanceof DataView);
  }
  function J(e5) {
    return "[object SharedArrayBuffer]" === m2(e5);
  }
  function _(e5) {
    return "undefined" != typeof SharedArrayBuffer && (J.working ? J(e5) : e5 instanceof SharedArrayBuffer);
  }
  function H(e5) {
    return O(e5, h3);
  }
  function Z(e5) {
    return O(e5, j);
  }
  function q(e5) {
    return O(e5, A);
  }
  function K(e5) {
    return s3 && O(e5, w);
  }
  function L(e5) {
    return p3 && O(e5, v2);
  }
  o$2.isArgumentsObject = f3, o$2.isGeneratorFunction = a3, o$2.isPromise = function(e5) {
    return "undefined" != typeof Promise && e5 instanceof Promise || null !== e5 && "object" == typeof e5 && "function" == typeof e5.then && "function" == typeof e5.catch;
  }, o$2.isArrayBufferView = function(e5) {
    return d2 && ArrayBuffer.isView ? ArrayBuffer.isView(e5) : S(e5) || R(e5);
  }, o$2.isTypedArray = S, o$2.isUint8Array = B, o$2.isUint8ClampedArray = k, o$2.isUint16Array = E, o$2.isUint32Array = D, o$2.isInt8Array = U, o$2.isInt16Array = P, o$2.isInt32Array = x, o$2.isFloat32Array = I, o$2.isFloat64Array = M, o$2.isBigInt64Array = z, o$2.isBigUint64Array = F, T2.working = "undefined" != typeof Map && T2(/* @__PURE__ */ new Map()), o$2.isMap = function(e5) {
    return "undefined" != typeof Map && (T2.working ? T2(e5) : e5 instanceof Map);
  }, N.working = "undefined" != typeof Set && N(/* @__PURE__ */ new Set()), o$2.isSet = function(e5) {
    return "undefined" != typeof Set && (N.working ? N(e5) : e5 instanceof Set);
  }, W.working = "undefined" != typeof WeakMap && W(/* @__PURE__ */ new WeakMap()), o$2.isWeakMap = function(e5) {
    return "undefined" != typeof WeakMap && (W.working ? W(e5) : e5 instanceof WeakMap);
  }, $.working = "undefined" != typeof WeakSet && $(/* @__PURE__ */ new WeakSet()), o$2.isWeakSet = function(e5) {
    return $(e5);
  }, C.working = "undefined" != typeof ArrayBuffer && C(new ArrayBuffer()), o$2.isArrayBuffer = V, G.working = "undefined" != typeof ArrayBuffer && "undefined" != typeof DataView && G(new DataView(new ArrayBuffer(1), 0, 1)), o$2.isDataView = R, J.working = "undefined" != typeof SharedArrayBuffer && J(new SharedArrayBuffer()), o$2.isSharedArrayBuffer = _, o$2.isAsyncFunction = function(e5) {
    return "[object AsyncFunction]" === m2(e5);
  }, o$2.isMapIterator = function(e5) {
    return "[object Map Iterator]" === m2(e5);
  }, o$2.isSetIterator = function(e5) {
    return "[object Set Iterator]" === m2(e5);
  }, o$2.isGeneratorObject = function(e5) {
    return "[object Generator]" === m2(e5);
  }, o$2.isWebAssemblyCompiledModule = function(e5) {
    return "[object WebAssembly.Module]" === m2(e5);
  }, o$2.isNumberObject = H, o$2.isStringObject = Z, o$2.isBooleanObject = q, o$2.isBigIntObject = K, o$2.isSymbolObject = L, o$2.isBoxedPrimitive = function(e5) {
    return H(e5) || Z(e5) || q(e5) || K(e5) || L(e5);
  }, o$2.isAnyArrayBuffer = function(e5) {
    return l$1 && (V(e5) || _(e5));
  }, ["isProxy", "isExternal", "isModuleNamespaceObject"].forEach(function(e5) {
    Object.defineProperty(o$2, e5, { enumerable: false, value: function() {
      throw new Error(e5 + " is not supported in userland");
    } });
  });
  var Q = "undefined" != typeof globalThis ? globalThis : "undefined" != typeof self ? self : global;
  var X = {};
  var Y = T;
  var ee = Object.getOwnPropertyDescriptors || function(e5) {
    for (var t5 = Object.keys(e5), r5 = {}, n4 = 0; n4 < t5.length; n4++)
      r5[t5[n4]] = Object.getOwnPropertyDescriptor(e5, t5[n4]);
    return r5;
  };
  var te = /%[sdj%]/g;
  X.format = function(e5) {
    if (!ge(e5)) {
      for (var t5 = [], r5 = 0; r5 < arguments.length; r5++)
        t5.push(oe(arguments[r5]));
      return t5.join(" ");
    }
    r5 = 1;
    for (var n4 = arguments, i4 = n4.length, o4 = String(e5).replace(te, function(e6) {
      if ("%%" === e6)
        return "%";
      if (r5 >= i4)
        return e6;
      switch (e6) {
        case "%s":
          return String(n4[r5++]);
        case "%d":
          return Number(n4[r5++]);
        case "%j":
          try {
            return JSON.stringify(n4[r5++]);
          } catch (e7) {
            return "[Circular]";
          }
        default:
          return e6;
      }
    }), u4 = n4[r5]; r5 < i4; u4 = n4[++r5])
      le(u4) || !he(u4) ? o4 += " " + u4 : o4 += " " + oe(u4);
    return o4;
  }, X.deprecate = function(e5, t5) {
    if (void 0 !== Y && true === Y.noDeprecation)
      return e5;
    if (void 0 === Y)
      return function() {
        return X.deprecate(e5, t5).apply(this || Q, arguments);
      };
    var r5 = false;
    return function() {
      if (!r5) {
        if (Y.throwDeprecation)
          throw new Error(t5);
        Y.traceDeprecation ? console.trace(t5) : console.error(t5), r5 = true;
      }
      return e5.apply(this || Q, arguments);
    };
  };
  var re = {};
  var ne = /^$/;
  if (Y.env.NODE_DEBUG) {
    ie = Y.env.NODE_DEBUG;
    ie = ie.replace(/[|\\{}()[\]^$+?.]/g, "\\$&").replace(/\*/g, ".*").replace(/,/g, "$|^").toUpperCase(), ne = new RegExp("^" + ie + "$", "i");
  }
  var ie;
  function oe(e5, t5) {
    var r5 = { seen: [], stylize: fe };
    return arguments.length >= 3 && (r5.depth = arguments[2]), arguments.length >= 4 && (r5.colors = arguments[3]), ye(t5) ? r5.showHidden = t5 : t5 && X._extend(r5, t5), be(r5.showHidden) && (r5.showHidden = false), be(r5.depth) && (r5.depth = 2), be(r5.colors) && (r5.colors = false), be(r5.customInspect) && (r5.customInspect = true), r5.colors && (r5.stylize = ue), ae(r5, e5, r5.depth);
  }
  function ue(e5, t5) {
    var r5 = oe.styles[t5];
    return r5 ? "\x1B[" + oe.colors[r5][0] + "m" + e5 + "\x1B[" + oe.colors[r5][1] + "m" : e5;
  }
  function fe(e5, t5) {
    return e5;
  }
  function ae(e5, t5, r5) {
    if (e5.customInspect && t5 && we(t5.inspect) && t5.inspect !== X.inspect && (!t5.constructor || t5.constructor.prototype !== t5)) {
      var n4 = t5.inspect(r5, e5);
      return ge(n4) || (n4 = ae(e5, n4, r5)), n4;
    }
    var i4 = function(e6, t6) {
      if (be(t6))
        return e6.stylize("undefined", "undefined");
      if (ge(t6)) {
        var r6 = "'" + JSON.stringify(t6).replace(/^"|"$/g, "").replace(/'/g, "\\'").replace(/\\"/g, '"') + "'";
        return e6.stylize(r6, "string");
      }
      if (de(t6))
        return e6.stylize("" + t6, "number");
      if (ye(t6))
        return e6.stylize("" + t6, "boolean");
      if (le(t6))
        return e6.stylize("null", "null");
    }(e5, t5);
    if (i4)
      return i4;
    var o4 = Object.keys(t5), u4 = function(e6) {
      var t6 = {};
      return e6.forEach(function(e7, r6) {
        t6[e7] = true;
      }), t6;
    }(o4);
    if (e5.showHidden && (o4 = Object.getOwnPropertyNames(t5)), Ae(t5) && (o4.indexOf("message") >= 0 || o4.indexOf("description") >= 0))
      return ce(t5);
    if (0 === o4.length) {
      if (we(t5)) {
        var f4 = t5.name ? ": " + t5.name : "";
        return e5.stylize("[Function" + f4 + "]", "special");
      }
      if (me(t5))
        return e5.stylize(RegExp.prototype.toString.call(t5), "regexp");
      if (je(t5))
        return e5.stylize(Date.prototype.toString.call(t5), "date");
      if (Ae(t5))
        return ce(t5);
    }
    var a4, c4 = "", s4 = false, p4 = ["{", "}"];
    (pe(t5) && (s4 = true, p4 = ["[", "]"]), we(t5)) && (c4 = " [Function" + (t5.name ? ": " + t5.name : "") + "]");
    return me(t5) && (c4 = " " + RegExp.prototype.toString.call(t5)), je(t5) && (c4 = " " + Date.prototype.toUTCString.call(t5)), Ae(t5) && (c4 = " " + ce(t5)), 0 !== o4.length || s4 && 0 != t5.length ? r5 < 0 ? me(t5) ? e5.stylize(RegExp.prototype.toString.call(t5), "regexp") : e5.stylize("[Object]", "special") : (e5.seen.push(t5), a4 = s4 ? function(e6, t6, r6, n5, i5) {
      for (var o5 = [], u5 = 0, f5 = t6.length; u5 < f5; ++u5)
        ke(t6, String(u5)) ? o5.push(se(e6, t6, r6, n5, String(u5), true)) : o5.push("");
      return i5.forEach(function(i6) {
        i6.match(/^\d+$/) || o5.push(se(e6, t6, r6, n5, i6, true));
      }), o5;
    }(e5, t5, r5, u4, o4) : o4.map(function(n5) {
      return se(e5, t5, r5, u4, n5, s4);
    }), e5.seen.pop(), function(e6, t6, r6) {
      var n5 = 0;
      if (e6.reduce(function(e7, t7) {
        return n5++, t7.indexOf("\n") >= 0 && n5++, e7 + t7.replace(/\u001b\[\d\d?m/g, "").length + 1;
      }, 0) > 60)
        return r6[0] + ("" === t6 ? "" : t6 + "\n ") + " " + e6.join(",\n  ") + " " + r6[1];
      return r6[0] + t6 + " " + e6.join(", ") + " " + r6[1];
    }(a4, c4, p4)) : p4[0] + c4 + p4[1];
  }
  function ce(e5) {
    return "[" + Error.prototype.toString.call(e5) + "]";
  }
  function se(e5, t5, r5, n4, i4, o4) {
    var u4, f4, a4;
    if ((a4 = Object.getOwnPropertyDescriptor(t5, i4) || { value: t5[i4] }).get ? f4 = a4.set ? e5.stylize("[Getter/Setter]", "special") : e5.stylize("[Getter]", "special") : a4.set && (f4 = e5.stylize("[Setter]", "special")), ke(n4, i4) || (u4 = "[" + i4 + "]"), f4 || (e5.seen.indexOf(a4.value) < 0 ? (f4 = le(r5) ? ae(e5, a4.value, null) : ae(e5, a4.value, r5 - 1)).indexOf("\n") > -1 && (f4 = o4 ? f4.split("\n").map(function(e6) {
      return "  " + e6;
    }).join("\n").substr(2) : "\n" + f4.split("\n").map(function(e6) {
      return "   " + e6;
    }).join("\n")) : f4 = e5.stylize("[Circular]", "special")), be(u4)) {
      if (o4 && i4.match(/^\d+$/))
        return f4;
      (u4 = JSON.stringify("" + i4)).match(/^"([a-zA-Z_][a-zA-Z_0-9]*)"$/) ? (u4 = u4.substr(1, u4.length - 2), u4 = e5.stylize(u4, "name")) : (u4 = u4.replace(/'/g, "\\'").replace(/\\"/g, '"').replace(/(^"|"$)/g, "'"), u4 = e5.stylize(u4, "string"));
    }
    return u4 + ": " + f4;
  }
  function pe(e5) {
    return Array.isArray(e5);
  }
  function ye(e5) {
    return "boolean" == typeof e5;
  }
  function le(e5) {
    return null === e5;
  }
  function de(e5) {
    return "number" == typeof e5;
  }
  function ge(e5) {
    return "string" == typeof e5;
  }
  function be(e5) {
    return void 0 === e5;
  }
  function me(e5) {
    return he(e5) && "[object RegExp]" === ve(e5);
  }
  function he(e5) {
    return "object" == typeof e5 && null !== e5;
  }
  function je(e5) {
    return he(e5) && "[object Date]" === ve(e5);
  }
  function Ae(e5) {
    return he(e5) && ("[object Error]" === ve(e5) || e5 instanceof Error);
  }
  function we(e5) {
    return "function" == typeof e5;
  }
  function ve(e5) {
    return Object.prototype.toString.call(e5);
  }
  function Oe(e5) {
    return e5 < 10 ? "0" + e5.toString(10) : e5.toString(10);
  }
  X.debuglog = function(e5) {
    if (e5 = e5.toUpperCase(), !re[e5])
      if (ne.test(e5)) {
        var t5 = Y.pid;
        re[e5] = function() {
          var r5 = X.format.apply(X, arguments);
          console.error("%s %d: %s", e5, t5, r5);
        };
      } else
        re[e5] = function() {
        };
    return re[e5];
  }, X.inspect = oe, oe.colors = { bold: [1, 22], italic: [3, 23], underline: [4, 24], inverse: [7, 27], white: [37, 39], grey: [90, 39], black: [30, 39], blue: [34, 39], cyan: [36, 39], green: [32, 39], magenta: [35, 39], red: [31, 39], yellow: [33, 39] }, oe.styles = { special: "cyan", number: "yellow", boolean: "yellow", undefined: "grey", null: "bold", string: "green", date: "magenta", regexp: "red" }, X.types = o$2, X.isArray = pe, X.isBoolean = ye, X.isNull = le, X.isNullOrUndefined = function(e5) {
    return null == e5;
  }, X.isNumber = de, X.isString = ge, X.isSymbol = function(e5) {
    return "symbol" == typeof e5;
  }, X.isUndefined = be, X.isRegExp = me, X.types.isRegExp = me, X.isObject = he, X.isDate = je, X.types.isDate = je, X.isError = Ae, X.types.isNativeError = Ae, X.isFunction = we, X.isPrimitive = function(e5) {
    return null === e5 || "boolean" == typeof e5 || "number" == typeof e5 || "string" == typeof e5 || "symbol" == typeof e5 || void 0 === e5;
  }, X.isBuffer = i$1;
  var Se = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function Be() {
    var e5 = /* @__PURE__ */ new Date(), t5 = [Oe(e5.getHours()), Oe(e5.getMinutes()), Oe(e5.getSeconds())].join(":");
    return [e5.getDate(), Se[e5.getMonth()], t5].join(" ");
  }
  function ke(e5, t5) {
    return Object.prototype.hasOwnProperty.call(e5, t5);
  }
  X.log = function() {
    console.log("%s - %s", Be(), X.format.apply(X, arguments));
  }, X.inherits = t$2, X._extend = function(e5, t5) {
    if (!t5 || !he(t5))
      return e5;
    for (var r5 = Object.keys(t5), n4 = r5.length; n4--; )
      e5[r5[n4]] = t5[r5[n4]];
    return e5;
  };
  var Ee = "undefined" != typeof Symbol ? Symbol("util.promisify.custom") : void 0;
  function De(e5, t5) {
    if (!e5) {
      var r5 = new Error("Promise was rejected with a falsy value");
      r5.reason = e5, e5 = r5;
    }
    return t5(e5);
  }
  X.promisify = function(e5) {
    if ("function" != typeof e5)
      throw new TypeError('The "original" argument must be of type Function');
    if (Ee && e5[Ee]) {
      var t5;
      if ("function" != typeof (t5 = e5[Ee]))
        throw new TypeError('The "util.promisify.custom" argument must be of type Function');
      return Object.defineProperty(t5, Ee, { value: t5, enumerable: false, writable: false, configurable: true }), t5;
    }
    function t5() {
      for (var t6, r5, n4 = new Promise(function(e6, n5) {
        t6 = e6, r5 = n5;
      }), i4 = [], o4 = 0; o4 < arguments.length; o4++)
        i4.push(arguments[o4]);
      i4.push(function(e6, n5) {
        e6 ? r5(e6) : t6(n5);
      });
      try {
        e5.apply(this || Q, i4);
      } catch (e6) {
        r5(e6);
      }
      return n4;
    }
    return Object.setPrototypeOf(t5, Object.getPrototypeOf(e5)), Ee && Object.defineProperty(t5, Ee, { value: t5, enumerable: false, writable: false, configurable: true }), Object.defineProperties(t5, ee(e5));
  }, X.promisify.custom = Ee, X.callbackify = function(e5) {
    if ("function" != typeof e5)
      throw new TypeError('The "original" argument must be of type Function');
    function t5() {
      for (var t6 = [], r5 = 0; r5 < arguments.length; r5++)
        t6.push(arguments[r5]);
      var n4 = t6.pop();
      if ("function" != typeof n4)
        throw new TypeError("The last argument must be of type Function");
      var i4 = this || Q, o4 = function() {
        return n4.apply(i4, arguments);
      };
      e5.apply(this || Q, t6).then(function(e6) {
        Y.nextTick(o4.bind(null, null, e6));
      }, function(e6) {
        Y.nextTick(De.bind(null, e6, o4));
      });
    }
    return Object.setPrototypeOf(t5, Object.getPrototypeOf(e5)), Object.defineProperties(t5, ee(e5)), t5;
  };

  // node_modules/@jspm/core/nodelibs/browser/assert.js
  function e4(e5, r5) {
    if (null == e5)
      throw new TypeError("Cannot convert first argument to object");
    for (var t5 = Object(e5), n4 = 1; n4 < arguments.length; n4++) {
      var o4 = arguments[n4];
      if (null != o4)
        for (var a4 = Object.keys(Object(o4)), l4 = 0, i4 = a4.length; l4 < i4; l4++) {
          var c4 = a4[l4], b3 = Object.getOwnPropertyDescriptor(o4, c4);
          void 0 !== b3 && b3.enumerable && (t5[c4] = o4[c4]);
        }
    }
    return t5;
  }
  var r4 = { assign: e4, polyfill: function() {
    Object.assign || Object.defineProperty(Object, "assign", { enumerable: false, configurable: true, writable: true, value: e4 });
  } };
  var t4;
  var e$12 = Object.prototype.toString;
  var r$12 = function(t5) {
    var r5 = e$12.call(t5), n4 = "[object Arguments]" === r5;
    return n4 || (n4 = "[object Array]" !== r5 && null !== t5 && "object" == typeof t5 && "number" == typeof t5.length && t5.length >= 0 && "[object Function]" === e$12.call(t5.callee)), n4;
  };
  if (!Object.keys) {
    n4 = Object.prototype.hasOwnProperty, o4 = Object.prototype.toString, c4 = r$12, l4 = Object.prototype.propertyIsEnumerable, i4 = !l4.call({ toString: null }, "toString"), a4 = l4.call(function() {
    }, "prototype"), u4 = ["toString", "toLocaleString", "valueOf", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable", "constructor"], f4 = function(t5) {
      var e5 = t5.constructor;
      return e5 && e5.prototype === t5;
    }, p4 = { $applicationCache: true, $console: true, $external: true, $frame: true, $frameElement: true, $frames: true, $innerHeight: true, $innerWidth: true, $onmozfullscreenchange: true, $onmozfullscreenerror: true, $outerHeight: true, $outerWidth: true, $pageXOffset: true, $pageYOffset: true, $parent: true, $scrollLeft: true, $scrollTop: true, $scrollX: true, $scrollY: true, $self: true, $webkitIndexedDB: true, $webkitStorageInfo: true, $window: true }, s4 = function() {
      if ("undefined" == typeof window)
        return false;
      for (var t5 in window)
        try {
          if (!p4["$" + t5] && n4.call(window, t5) && null !== window[t5] && "object" == typeof window[t5])
            try {
              f4(window[t5]);
            } catch (t6) {
              return true;
            }
        } catch (t6) {
          return true;
        }
      return false;
    }();
    t4 = function(t5) {
      var e5 = null !== t5 && "object" == typeof t5, r5 = "[object Function]" === o4.call(t5), l5 = c4(t5), p5 = e5 && "[object String]" === o4.call(t5), y4 = [];
      if (!e5 && !r5 && !l5)
        throw new TypeError("Object.keys called on a non-object");
      var b3 = a4 && r5;
      if (p5 && t5.length > 0 && !n4.call(t5, 0))
        for (var g3 = 0; g3 < t5.length; ++g3)
          y4.push(String(g3));
      if (l5 && t5.length > 0)
        for (var h5 = 0; h5 < t5.length; ++h5)
          y4.push(String(h5));
      else
        for (var $3 in t5)
          b3 && "prototype" === $3 || !n4.call(t5, $3) || y4.push(String($3));
      if (i4)
        for (var j3 = function(t6) {
          if ("undefined" == typeof window || !s4)
            return f4(t6);
          try {
            return f4(t6);
          } catch (t7) {
            return false;
          }
        }(t5), w3 = 0; w3 < u4.length; ++w3)
          j3 && "constructor" === u4[w3] || !n4.call(t5, u4[w3]) || y4.push(u4[w3]);
      return y4;
    };
  }
  var n4;
  var o4;
  var c4;
  var l4;
  var i4;
  var a4;
  var u4;
  var f4;
  var p4;
  var s4;
  var y3 = t4;
  var b2 = Array.prototype.slice;
  var g2 = r$12;
  var h4 = Object.keys;
  var $2 = h4 ? function(t5) {
    return h4(t5);
  } : y3;
  var j2 = Object.keys;
  $2.shim = function() {
    Object.keys ? function() {
      var t5 = Object.keys(arguments);
      return t5 && t5.length === arguments.length;
    }(1, 2) || (Object.keys = function(t5) {
      return g2(t5) ? j2(b2.call(t5)) : j2(t5);
    }) : Object.keys = $2;
    return Object.keys || $2;
  };
  var w2 = $2;
  var r$2 = w2;
  var e$2 = "function" == typeof Symbol && "symbol" == typeof Symbol("foo");
  var o$12 = Object.prototype.toString;
  var n$12 = Array.prototype.concat;
  var a$1 = Object.defineProperty;
  var c$12 = a$1 && function() {
    var t5 = {};
    try {
      for (var r5 in a$1(t5, "x", { enumerable: false, value: t5 }), t5)
        return false;
      return t5.x === t5;
    } catch (t6) {
      return false;
    }
  }();
  var l$12 = function(t5, r5, e5, n4) {
    var l4;
    (!(r5 in t5) || "function" == typeof (l4 = n4) && "[object Function]" === o$12.call(l4) && n4()) && (c$12 ? a$1(t5, r5, { configurable: true, enumerable: false, value: e5, writable: true }) : t5[r5] = e5);
  };
  var u$12 = function(t5, o4) {
    var a4 = arguments.length > 2 ? arguments[2] : {}, c4 = r$2(o4);
    e$2 && (c4 = n$12.call(c4, Object.getOwnPropertySymbols(o4)));
    for (var u4 = 0; u4 < c4.length; u4 += 1)
      l$12(t5, c4[u4], o4[c4[u4]], a4[c4[u4]]);
  };
  u$12.supportsDescriptors = !!c$12;
  var f$1 = u$12;
  var t$12 = function() {
    if ("function" != typeof Symbol || "function" != typeof Object.getOwnPropertySymbols)
      return false;
    if ("symbol" == typeof Symbol.iterator)
      return true;
    var t5 = {}, e5 = Symbol("test"), r5 = Object(e5);
    if ("string" == typeof e5)
      return false;
    if ("[object Symbol]" !== Object.prototype.toString.call(e5))
      return false;
    if ("[object Symbol]" !== Object.prototype.toString.call(r5))
      return false;
    for (e5 in t5[e5] = 42, t5)
      return false;
    if ("function" == typeof Object.keys && 0 !== Object.keys(t5).length)
      return false;
    if ("function" == typeof Object.getOwnPropertyNames && 0 !== Object.getOwnPropertyNames(t5).length)
      return false;
    var o4 = Object.getOwnPropertySymbols(t5);
    if (1 !== o4.length || o4[0] !== e5)
      return false;
    if (!Object.prototype.propertyIsEnumerable.call(t5, e5))
      return false;
    if ("function" == typeof Object.getOwnPropertyDescriptor) {
      var n4 = Object.getOwnPropertyDescriptor(t5, e5);
      if (42 !== n4.value || true !== n4.enumerable)
        return false;
    }
    return true;
  };
  var f$2 = ("undefined" != typeof globalThis ? globalThis : "undefined" != typeof self ? self : global).Symbol;
  var e$3 = t$12;
  var l$2 = function() {
    return "function" == typeof f$2 && ("function" == typeof Symbol && ("symbol" == typeof f$2("foo") && ("symbol" == typeof Symbol("bar") && e$3())));
  };
  var t$22 = "Function.prototype.bind called on incompatible ";
  var n$2 = Array.prototype.slice;
  var o$22 = Object.prototype.toString;
  var r$3 = function(r5) {
    var e5 = this;
    if ("function" != typeof e5 || "[object Function]" !== o$22.call(e5))
      throw new TypeError(t$22 + e5);
    for (var p4, i4 = n$2.call(arguments, 1), c4 = function() {
      if (this instanceof p4) {
        var t5 = e5.apply(this, i4.concat(n$2.call(arguments)));
        return Object(t5) === t5 ? t5 : this;
      }
      return e5.apply(r5, i4.concat(n$2.call(arguments)));
    }, a4 = Math.max(0, e5.length - i4.length), l4 = [], u4 = 0; u4 < a4; u4++)
      l4.push("$" + u4);
    if (p4 = Function("binder", "return function (" + l4.join(",") + "){ return binder.apply(this,arguments); }")(c4), e5.prototype) {
      var y4 = function() {
      };
      y4.prototype = e5.prototype, p4.prototype = new y4(), y4.prototype = null;
    }
    return p4;
  };
  var e$4 = Function.prototype.bind || r$3;
  var o$3 = TypeError;
  var t$3 = Object.getOwnPropertyDescriptor;
  if (t$3)
    try {
      t$3({}, "");
    } catch (r5) {
      t$3 = null;
    }
  var n$3 = function() {
    throw new o$3();
  };
  var y$1 = t$3 ? function() {
    try {
      return arguments.callee, n$3;
    } catch (r5) {
      try {
        return t$3(arguments, "callee").get;
      } catch (r6) {
        return n$3;
      }
    }
  }() : n$3;
  var a$2 = l$2();
  var i$12 = Object.getPrototypeOf || function(r5) {
    return r5.__proto__;
  };
  var d3 = "undefined" == typeof Uint8Array ? void 0 : i$12(Uint8Array);
  var f$3 = { "%Array%": Array, "%ArrayBuffer%": "undefined" == typeof ArrayBuffer ? void 0 : ArrayBuffer, "%ArrayBufferPrototype%": "undefined" == typeof ArrayBuffer ? void 0 : ArrayBuffer.prototype, "%ArrayIteratorPrototype%": a$2 ? i$12([][Symbol.iterator]()) : void 0, "%ArrayPrototype%": Array.prototype, "%ArrayProto_entries%": Array.prototype.entries, "%ArrayProto_forEach%": Array.prototype.forEach, "%ArrayProto_keys%": Array.prototype.keys, "%ArrayProto_values%": Array.prototype.values, "%AsyncFromSyncIteratorPrototype%": void 0, "%AsyncFunction%": void 0, "%AsyncFunctionPrototype%": void 0, "%AsyncGenerator%": void 0, "%AsyncGeneratorFunction%": void 0, "%AsyncGeneratorPrototype%": void 0, "%AsyncIteratorPrototype%": void 0, "%Atomics%": "undefined" == typeof Atomics ? void 0 : Atomics, "%Boolean%": Boolean, "%BooleanPrototype%": Boolean.prototype, "%DataView%": "undefined" == typeof DataView ? void 0 : DataView, "%DataViewPrototype%": "undefined" == typeof DataView ? void 0 : DataView.prototype, "%Date%": Date, "%DatePrototype%": Date.prototype, "%decodeURI%": decodeURI, "%decodeURIComponent%": decodeURIComponent, "%encodeURI%": encodeURI, "%encodeURIComponent%": encodeURIComponent, "%Error%": Error, "%ErrorPrototype%": Error.prototype, "%eval%": eval, "%EvalError%": EvalError, "%EvalErrorPrototype%": EvalError.prototype, "%Float32Array%": "undefined" == typeof Float32Array ? void 0 : Float32Array, "%Float32ArrayPrototype%": "undefined" == typeof Float32Array ? void 0 : Float32Array.prototype, "%Float64Array%": "undefined" == typeof Float64Array ? void 0 : Float64Array, "%Float64ArrayPrototype%": "undefined" == typeof Float64Array ? void 0 : Float64Array.prototype, "%Function%": Function, "%FunctionPrototype%": Function.prototype, "%Generator%": void 0, "%GeneratorFunction%": void 0, "%GeneratorPrototype%": void 0, "%Int8Array%": "undefined" == typeof Int8Array ? void 0 : Int8Array, "%Int8ArrayPrototype%": "undefined" == typeof Int8Array ? void 0 : Int8Array.prototype, "%Int16Array%": "undefined" == typeof Int16Array ? void 0 : Int16Array, "%Int16ArrayPrototype%": "undefined" == typeof Int16Array ? void 0 : Int8Array.prototype, "%Int32Array%": "undefined" == typeof Int32Array ? void 0 : Int32Array, "%Int32ArrayPrototype%": "undefined" == typeof Int32Array ? void 0 : Int32Array.prototype, "%isFinite%": isFinite, "%isNaN%": isNaN, "%IteratorPrototype%": a$2 ? i$12(i$12([][Symbol.iterator]())) : void 0, "%JSON%": "object" == typeof JSON ? JSON : void 0, "%JSONParse%": "object" == typeof JSON ? JSON.parse : void 0, "%Map%": "undefined" == typeof Map ? void 0 : Map, "%MapIteratorPrototype%": "undefined" != typeof Map && a$2 ? i$12((/* @__PURE__ */ new Map())[Symbol.iterator]()) : void 0, "%MapPrototype%": "undefined" == typeof Map ? void 0 : Map.prototype, "%Math%": Math, "%Number%": Number, "%NumberPrototype%": Number.prototype, "%Object%": Object, "%ObjectPrototype%": Object.prototype, "%ObjProto_toString%": Object.prototype.toString, "%ObjProto_valueOf%": Object.prototype.valueOf, "%parseFloat%": parseFloat, "%parseInt%": parseInt, "%Promise%": "undefined" == typeof Promise ? void 0 : Promise, "%PromisePrototype%": "undefined" == typeof Promise ? void 0 : Promise.prototype, "%PromiseProto_then%": "undefined" == typeof Promise ? void 0 : Promise.prototype.then, "%Promise_all%": "undefined" == typeof Promise ? void 0 : Promise.all, "%Promise_reject%": "undefined" == typeof Promise ? void 0 : Promise.reject, "%Promise_resolve%": "undefined" == typeof Promise ? void 0 : Promise.resolve, "%Proxy%": "undefined" == typeof Proxy ? void 0 : Proxy, "%RangeError%": RangeError, "%RangeErrorPrototype%": RangeError.prototype, "%ReferenceError%": ReferenceError, "%ReferenceErrorPrototype%": ReferenceError.prototype, "%Reflect%": "undefined" == typeof Reflect ? void 0 : Reflect, "%RegExp%": RegExp, "%RegExpPrototype%": RegExp.prototype, "%Set%": "undefined" == typeof Set ? void 0 : Set, "%SetIteratorPrototype%": "undefined" != typeof Set && a$2 ? i$12((/* @__PURE__ */ new Set())[Symbol.iterator]()) : void 0, "%SetPrototype%": "undefined" == typeof Set ? void 0 : Set.prototype, "%SharedArrayBuffer%": "undefined" == typeof SharedArrayBuffer ? void 0 : SharedArrayBuffer, "%SharedArrayBufferPrototype%": "undefined" == typeof SharedArrayBuffer ? void 0 : SharedArrayBuffer.prototype, "%String%": String, "%StringIteratorPrototype%": a$2 ? i$12(""[Symbol.iterator]()) : void 0, "%StringPrototype%": String.prototype, "%Symbol%": a$2 ? Symbol : void 0, "%SymbolPrototype%": a$2 ? Symbol.prototype : void 0, "%SyntaxError%": SyntaxError, "%SyntaxErrorPrototype%": SyntaxError.prototype, "%ThrowTypeError%": y$1, "%TypedArray%": d3, "%TypedArrayPrototype%": d3 ? d3.prototype : void 0, "%TypeError%": o$3, "%TypeErrorPrototype%": o$3.prototype, "%Uint8Array%": "undefined" == typeof Uint8Array ? void 0 : Uint8Array, "%Uint8ArrayPrototype%": "undefined" == typeof Uint8Array ? void 0 : Uint8Array.prototype, "%Uint8ClampedArray%": "undefined" == typeof Uint8ClampedArray ? void 0 : Uint8ClampedArray, "%Uint8ClampedArrayPrototype%": "undefined" == typeof Uint8ClampedArray ? void 0 : Uint8ClampedArray.prototype, "%Uint16Array%": "undefined" == typeof Uint16Array ? void 0 : Uint16Array, "%Uint16ArrayPrototype%": "undefined" == typeof Uint16Array ? void 0 : Uint16Array.prototype, "%Uint32Array%": "undefined" == typeof Uint32Array ? void 0 : Uint32Array, "%Uint32ArrayPrototype%": "undefined" == typeof Uint32Array ? void 0 : Uint32Array.prototype, "%URIError%": URIError, "%URIErrorPrototype%": URIError.prototype, "%WeakMap%": "undefined" == typeof WeakMap ? void 0 : WeakMap, "%WeakMapPrototype%": "undefined" == typeof WeakMap ? void 0 : WeakMap.prototype, "%WeakSet%": "undefined" == typeof WeakSet ? void 0 : WeakSet, "%WeakSetPrototype%": "undefined" == typeof WeakSet ? void 0 : WeakSet.prototype };
  var u$2 = e$4.call(Function.call, String.prototype.replace);
  var A2 = /[^%.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|%$))/g;
  var l$3 = /\\(\\)?/g;
  var v3 = function(r5) {
    var e5 = [];
    return u$2(r5, A2, function(r6, o4, t5, n4) {
      e5[e5.length] = t5 ? u$2(n4, l$3, "$1") : o4 || r6;
    }), e5;
  };
  var P2 = function(r5, e5) {
    if (!(r5 in f$3))
      throw new SyntaxError("intrinsic " + r5 + " does not exist!");
    if (void 0 === f$3[r5] && !e5)
      throw new o$3("intrinsic " + r5 + " exists, but is not available. Please file an issue!");
    return f$3[r5];
  };
  var c$2 = function(r5, e5) {
    if ("string" != typeof r5 || 0 === r5.length)
      throw new TypeError("intrinsic name must be a non-empty string");
    if (arguments.length > 1 && "boolean" != typeof e5)
      throw new TypeError('"allowMissing" argument must be a boolean');
    for (var n4 = v3(r5), y4 = P2("%" + (n4.length > 0 ? n4[0] : "") + "%", e5), a4 = 1; a4 < n4.length; a4 += 1)
      if (null != y4)
        if (t$3 && a4 + 1 >= n4.length) {
          var i4 = t$3(y4, n4[a4]);
          if (!e5 && !(n4[a4] in y4))
            throw new o$3("base intrinsic for " + r5 + " exists, but the property is not available.");
          y4 = i4 ? i4.get || i4.value : y4[n4[a4]];
        } else
          y4 = y4[n4[a4]];
    return y4;
  };
  var t$4;
  var p$1 = e$4;
  var o$4 = c$2("%Function%");
  var i$2 = o$4.apply;
  var a$3 = o$4.call;
  (t$4 = function() {
    return p$1.apply(a$3, arguments);
  }).apply = function() {
    return p$1.apply(i$2, arguments);
  };
  var l$4 = t$4;
  var r$4;
  var n$4;
  var i$3 = function(t5) {
    return t5 != t5;
  };
  var o$5 = (r$4 = function(t5, e5) {
    return 0 === t5 && 0 === e5 ? 1 / t5 == 1 / e5 : t5 === e5 || !(!i$3(t5) || !i$3(e5));
  }, r$4);
  var c$3 = (n$4 = function() {
    return "function" == typeof Object.is ? Object.is : o$5;
  }, n$4);
  var f$4 = f$1;
  var u$3 = f$1;
  var s$1 = r$4;
  var a$4 = n$4;
  var l$5 = function() {
    var t5 = c$3();
    return f$4(Object, { is: t5 }, { is: function() {
      return Object.is !== t5;
    } }), t5;
  };
  var p$2 = l$4(a$4(), Object);
  u$3(p$2, { getPolyfill: a$4, implementation: s$1, shim: l$5 });
  var m3 = p$2;
  N2 = function(r5) {
    return r5 != r5;
  };
  var N2;
  var e$5;
  var i$4 = N2;
  var n$5 = (e$5 = function() {
    return Number.isNaN && Number.isNaN(NaN) && !Number.isNaN("a") ? Number.isNaN : i$4;
  }, f$1);
  var t$5 = e$5;
  var u$4 = f$1;
  var a$5 = N2;
  var m$1 = e$5;
  var o$6 = function() {
    var r5 = t$5();
    return n$5(Number, { isNaN: r5 }, { isNaN: function() {
      return Number.isNaN !== r5;
    } }), r5;
  };
  var s$2 = m$1();
  u$4(s$2, { getPolyfill: m$1, implementation: a$5, shim: o$6 });
  var f$5 = s$2;
  var c$4 = {};
  var a$6 = false;
  function i$5() {
    if (a$6)
      return c$4;
    function e5(t5) {
      return (e5 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(t6) {
        return typeof t6;
      } : function(t6) {
        return t6 && "function" == typeof Symbol && t6.constructor === Symbol && t6 !== Symbol.prototype ? "symbol" : typeof t6;
      })(t5);
    }
    function n4(t5, n5) {
      return !n5 || "object" !== e5(n5) && "function" != typeof n5 ? function(t6) {
        if (void 0 === t6)
          throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
        return t6;
      }(t5) : n5;
    }
    function r5(t5) {
      return (r5 = Object.setPrototypeOf ? Object.getPrototypeOf : function(t6) {
        return t6.__proto__ || Object.getPrototypeOf(t6);
      })(t5);
    }
    function o4(t5, e6) {
      return (o4 = Object.setPrototypeOf || function(t6, e7) {
        return t6.__proto__ = e7, t6;
      })(t5, e6);
    }
    a$6 = true;
    var i4, u4, l4 = {};
    function f4(t5, e6, c4) {
      c4 || (c4 = Error);
      var a4 = function(c5) {
        function a5(o5, c6, i5) {
          var u5;
          return !function(t6, e7) {
            if (!(t6 instanceof e7))
              throw new TypeError("Cannot call a class as a function");
          }(this, a5), (u5 = n4(this, r5(a5).call(this, function(t6, n5, r6) {
            return "string" == typeof e6 ? e6 : e6(t6, n5, r6);
          }(o5, c6, i5)))).code = t5, u5;
        }
        return !function(t6, e7) {
          if ("function" != typeof e7 && null !== e7)
            throw new TypeError("Super expression must either be null or a function");
          t6.prototype = Object.create(e7 && e7.prototype, { constructor: { value: t6, writable: true, configurable: true } }), e7 && o4(t6, e7);
        }(a5, c5), a5;
      }(c4);
      l4[t5] = a4;
    }
    function s4(t5, e6) {
      if (Array.isArray(t5)) {
        var n5 = t5.length;
        return t5 = t5.map(function(t6) {
          return String(t6);
        }), n5 > 2 ? "one of ".concat(e6, " ").concat(t5.slice(0, n5 - 1).join(", "), ", or ") + t5[n5 - 1] : 2 === n5 ? "one of ".concat(e6, " ").concat(t5[0], " or ").concat(t5[1]) : "of ".concat(e6, " ").concat(t5[0]);
      }
      return "of ".concat(e6, " ").concat(String(t5));
    }
    return f4("ERR_AMBIGUOUS_ARGUMENT", 'The "%s" argument is ambiguous. %s', TypeError), f4("ERR_INVALID_ARG_TYPE", function(t5, n5, r6) {
      var o5, c4, u5;
      if (void 0 === i4 && (i4 = tt()), i4("string" == typeof t5, "'name' must be a string"), "string" == typeof n5 && (c4 = "not ", n5.substr(0, c4.length) === c4) ? (o5 = "must not be", n5 = n5.replace(/^not /, "")) : o5 = "must be", function(t6, e6, n6) {
        return (void 0 === n6 || n6 > t6.length) && (n6 = t6.length), t6.substring(n6 - e6.length, n6) === e6;
      }(t5, " argument"))
        u5 = "The ".concat(t5, " ").concat(o5, " ").concat(s4(n5, "type"));
      else {
        var l5 = function(t6, e6, n6) {
          return "number" != typeof n6 && (n6 = 0), !(n6 + e6.length > t6.length) && -1 !== t6.indexOf(e6, n6);
        }(t5, ".") ? "property" : "argument";
        u5 = 'The "'.concat(t5, '" ').concat(l5, " ").concat(o5, " ").concat(s4(n5, "type"));
      }
      return u5 += ". Received type ".concat(e5(r6));
    }, TypeError), f4("ERR_INVALID_ARG_VALUE", function(e6, n5) {
      var r6 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : "is invalid";
      void 0 === u4 && (u4 = X);
      var o5 = u4.inspect(n5);
      return o5.length > 128 && (o5 = "".concat(o5.slice(0, 128), "...")), "The argument '".concat(e6, "' ").concat(r6, ". Received ").concat(o5);
    }, TypeError), f4("ERR_INVALID_RETURN_VALUE", function(t5, n5, r6) {
      var o5;
      return o5 = r6 && r6.constructor && r6.constructor.name ? "instance of ".concat(r6.constructor.name) : "type ".concat(e5(r6)), "Expected ".concat(t5, ' to be returned from the "').concat(n5, '"') + " function but got ".concat(o5, ".");
    }, TypeError), f4("ERR_MISSING_ARGS", function() {
      for (var t5 = arguments.length, e6 = new Array(t5), n5 = 0; n5 < t5; n5++)
        e6[n5] = arguments[n5];
      void 0 === i4 && (i4 = tt()), i4(e6.length > 0, "At least one arg needs to be specified");
      var r6 = "The ", o5 = e6.length;
      switch (e6 = e6.map(function(t6) {
        return '"'.concat(t6, '"');
      }), o5) {
        case 1:
          r6 += "".concat(e6[0], " argument");
          break;
        case 2:
          r6 += "".concat(e6[0], " and ").concat(e6[1], " arguments");
          break;
        default:
          r6 += e6.slice(0, o5 - 1).join(", "), r6 += ", and ".concat(e6[o5 - 1], " arguments");
      }
      return "".concat(r6, " must be specified");
    }, TypeError), c$4.codes = l4, c$4;
  }
  var u$5 = {};
  var l$6 = false;
  function f$6() {
    if (l$6)
      return u$5;
    l$6 = true;
    var n4 = T;
    function r5(t5, e5, n5) {
      return e5 in t5 ? Object.defineProperty(t5, e5, { value: n5, enumerable: true, configurable: true, writable: true }) : t5[e5] = n5, t5;
    }
    function o4(t5, e5) {
      for (var n5 = 0; n5 < e5.length; n5++) {
        var r6 = e5[n5];
        r6.enumerable = r6.enumerable || false, r6.configurable = true, "value" in r6 && (r6.writable = true), Object.defineProperty(t5, r6.key, r6);
      }
    }
    function c4(t5, e5) {
      return !e5 || "object" !== y4(e5) && "function" != typeof e5 ? a4(t5) : e5;
    }
    function a4(t5) {
      if (void 0 === t5)
        throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
      return t5;
    }
    function f4(t5) {
      var e5 = "function" == typeof Map ? /* @__PURE__ */ new Map() : void 0;
      return (f4 = function(t6) {
        if (null === t6 || (n5 = t6, -1 === Function.toString.call(n5).indexOf("[native code]")))
          return t6;
        var n5;
        if ("function" != typeof t6)
          throw new TypeError("Super expression must either be null or a function");
        if (void 0 !== e5) {
          if (e5.has(t6))
            return e5.get(t6);
          e5.set(t6, r6);
        }
        function r6() {
          return p4(t6, arguments, h5(this).constructor);
        }
        return r6.prototype = Object.create(t6.prototype, { constructor: { value: r6, enumerable: false, writable: true, configurable: true } }), g3(r6, t6);
      })(t5);
    }
    function s4() {
      if ("undefined" == typeof Reflect || !Reflect.construct)
        return false;
      if (Reflect.construct.sham)
        return false;
      if ("function" == typeof Proxy)
        return true;
      try {
        return Date.prototype.toString.call(Reflect.construct(Date, [], function() {
        })), true;
      } catch (t5) {
        return false;
      }
    }
    function p4(t5, e5, n5) {
      return (p4 = s4() ? Reflect.construct : function(t6, e6, n6) {
        var r6 = [null];
        r6.push.apply(r6, e6);
        var o5 = new (Function.bind.apply(t6, r6))();
        return n6 && g3(o5, n6.prototype), o5;
      }).apply(null, arguments);
    }
    function g3(t5, e5) {
      return (g3 = Object.setPrototypeOf || function(t6, e6) {
        return t6.__proto__ = e6, t6;
      })(t5, e5);
    }
    function h5(t5) {
      return (h5 = Object.setPrototypeOf ? Object.getPrototypeOf : function(t6) {
        return t6.__proto__ || Object.getPrototypeOf(t6);
      })(t5);
    }
    function y4(t5) {
      return (y4 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(t6) {
        return typeof t6;
      } : function(t6) {
        return t6 && "function" == typeof Symbol && t6.constructor === Symbol && t6 !== Symbol.prototype ? "symbol" : typeof t6;
      })(t5);
    }
    var b3 = X.inspect, v4 = i$5().codes.ERR_INVALID_ARG_TYPE;
    function d4(t5, e5, n5) {
      return (void 0 === n5 || n5 > t5.length) && (n5 = t5.length), t5.substring(n5 - e5.length, n5) === e5;
    }
    var m4 = "", E3 = "", w3 = "", S3 = "", j3 = { deepStrictEqual: "Expected values to be strictly deep-equal:", strictEqual: "Expected values to be strictly equal:", strictEqualObject: 'Expected "actual" to be reference-equal to "expected":', deepEqual: "Expected values to be loosely deep-equal:", equal: "Expected values to be loosely equal:", notDeepStrictEqual: 'Expected "actual" not to be strictly deep-equal to:', notStrictEqual: 'Expected "actual" to be strictly unequal to:', notStrictEqualObject: 'Expected "actual" not to be reference-equal to "expected":', notDeepEqual: 'Expected "actual" not to be loosely deep-equal to:', notEqual: 'Expected "actual" to be loosely unequal to:', notIdentical: "Values identical but not reference-equal:" };
    function O3(t5) {
      var e5 = Object.keys(t5), n5 = Object.create(Object.getPrototypeOf(t5));
      return e5.forEach(function(e6) {
        n5[e6] = t5[e6];
      }), Object.defineProperty(n5, "message", { value: t5.message }), n5;
    }
    function x3(t5) {
      return b3(t5, { compact: false, customInspect: false, depth: 1e3, maxArrayLength: 1 / 0, showHidden: false, breakLength: 1 / 0, showProxy: false, sorted: true, getters: true });
    }
    function q3(t5, e5, r6) {
      var o5 = "", c5 = "", a5 = 0, i4 = "", u4 = false, l4 = x3(t5), f5 = l4.split("\n"), s5 = x3(e5).split("\n"), p5 = 0, g4 = "";
      if ("strictEqual" === r6 && "object" === y4(t5) && "object" === y4(e5) && null !== t5 && null !== e5 && (r6 = "strictEqualObject"), 1 === f5.length && 1 === s5.length && f5[0] !== s5[0]) {
        var h6 = f5[0].length + s5[0].length;
        if (h6 <= 10) {
          if (!("object" === y4(t5) && null !== t5 || "object" === y4(e5) && null !== e5 || 0 === t5 && 0 === e5))
            return "".concat(j3[r6], "\n\n") + "".concat(f5[0], " !== ").concat(s5[0], "\n");
        } else if ("strictEqualObject" !== r6) {
          if (h6 < (n4.stderr && n4.stderr.isTTY ? n4.stderr.columns : 80)) {
            for (; f5[0][p5] === s5[0][p5]; )
              p5++;
            p5 > 2 && (g4 = "\n  ".concat(function(t6, e6) {
              if (e6 = Math.floor(e6), 0 == t6.length || 0 == e6)
                return "";
              var n5 = t6.length * e6;
              for (e6 = Math.floor(Math.log(e6) / Math.log(2)); e6; )
                t6 += t6, e6--;
              return t6 += t6.substring(0, n5 - t6.length);
            }(" ", p5), "^"), p5 = 0);
          }
        }
      }
      for (var b4 = f5[f5.length - 1], v5 = s5[s5.length - 1]; b4 === v5 && (p5++ < 2 ? i4 = "\n  ".concat(b4).concat(i4) : o5 = b4, f5.pop(), s5.pop(), 0 !== f5.length && 0 !== s5.length); )
        b4 = f5[f5.length - 1], v5 = s5[s5.length - 1];
      var O4 = Math.max(f5.length, s5.length);
      if (0 === O4) {
        var q4 = l4.split("\n");
        if (q4.length > 30)
          for (q4[26] = "".concat(m4, "...").concat(S3); q4.length > 27; )
            q4.pop();
        return "".concat(j3.notIdentical, "\n\n").concat(q4.join("\n"), "\n");
      }
      p5 > 3 && (i4 = "\n".concat(m4, "...").concat(S3).concat(i4), u4 = true), "" !== o5 && (i4 = "\n  ".concat(o5).concat(i4), o5 = "");
      var R4 = 0, A3 = j3[r6] + "\n".concat(E3, "+ actual").concat(S3, " ").concat(w3, "- expected").concat(S3), k3 = " ".concat(m4, "...").concat(S3, " Lines skipped");
      for (p5 = 0; p5 < O4; p5++) {
        var _3 = p5 - a5;
        if (f5.length < p5 + 1)
          _3 > 1 && p5 > 2 && (_3 > 4 ? (c5 += "\n".concat(m4, "...").concat(S3), u4 = true) : _3 > 3 && (c5 += "\n  ".concat(s5[p5 - 2]), R4++), c5 += "\n  ".concat(s5[p5 - 1]), R4++), a5 = p5, o5 += "\n".concat(w3, "-").concat(S3, " ").concat(s5[p5]), R4++;
        else if (s5.length < p5 + 1)
          _3 > 1 && p5 > 2 && (_3 > 4 ? (c5 += "\n".concat(m4, "...").concat(S3), u4 = true) : _3 > 3 && (c5 += "\n  ".concat(f5[p5 - 2]), R4++), c5 += "\n  ".concat(f5[p5 - 1]), R4++), a5 = p5, c5 += "\n".concat(E3, "+").concat(S3, " ").concat(f5[p5]), R4++;
        else {
          var T4 = s5[p5], P3 = f5[p5], I3 = P3 !== T4 && (!d4(P3, ",") || P3.slice(0, -1) !== T4);
          I3 && d4(T4, ",") && T4.slice(0, -1) === P3 && (I3 = false, P3 += ","), I3 ? (_3 > 1 && p5 > 2 && (_3 > 4 ? (c5 += "\n".concat(m4, "...").concat(S3), u4 = true) : _3 > 3 && (c5 += "\n  ".concat(f5[p5 - 2]), R4++), c5 += "\n  ".concat(f5[p5 - 1]), R4++), a5 = p5, c5 += "\n".concat(E3, "+").concat(S3, " ").concat(P3), o5 += "\n".concat(w3, "-").concat(S3, " ").concat(T4), R4 += 2) : (c5 += o5, o5 = "", 1 !== _3 && 0 !== p5 || (c5 += "\n  ".concat(P3), R4++));
        }
        if (R4 > 20 && p5 < O4 - 2)
          return "".concat(A3).concat(k3, "\n").concat(c5, "\n").concat(m4, "...").concat(S3).concat(o5, "\n") + "".concat(m4, "...").concat(S3);
      }
      return "".concat(A3).concat(u4 ? k3 : "", "\n").concat(c5).concat(o5).concat(i4).concat(g4);
    }
    var R3 = function(t5) {
      function e5(t6) {
        var r6;
        if (!function(t7, e6) {
          if (!(t7 instanceof e6))
            throw new TypeError("Cannot call a class as a function");
        }(this, e5), "object" !== y4(t6) || null === t6)
          throw new v4("options", "Object", t6);
        var o5 = t6.message, i5 = t6.operator, u5 = t6.stackStartFn, l4 = t6.actual, f5 = t6.expected, s5 = Error.stackTraceLimit;
        if (Error.stackTraceLimit = 0, null != o5)
          r6 = c4(this, h5(e5).call(this, String(o5)));
        else if (n4.stderr && n4.stderr.isTTY && (n4.stderr && n4.stderr.getColorDepth && 1 !== n4.stderr.getColorDepth() ? (m4 = "\x1B[34m", E3 = "\x1B[32m", S3 = "\x1B[39m", w3 = "\x1B[31m") : (m4 = "", E3 = "", S3 = "", w3 = "")), "object" === y4(l4) && null !== l4 && "object" === y4(f5) && null !== f5 && "stack" in l4 && l4 instanceof Error && "stack" in f5 && f5 instanceof Error && (l4 = O3(l4), f5 = O3(f5)), "deepStrictEqual" === i5 || "strictEqual" === i5)
          r6 = c4(this, h5(e5).call(this, q3(l4, f5, i5)));
        else if ("notDeepStrictEqual" === i5 || "notStrictEqual" === i5) {
          var p5 = j3[i5], g4 = x3(l4).split("\n");
          if ("notStrictEqual" === i5 && "object" === y4(l4) && null !== l4 && (p5 = j3.notStrictEqualObject), g4.length > 30)
            for (g4[26] = "".concat(m4, "...").concat(S3); g4.length > 27; )
              g4.pop();
          r6 = 1 === g4.length ? c4(this, h5(e5).call(this, "".concat(p5, " ").concat(g4[0]))) : c4(this, h5(e5).call(this, "".concat(p5, "\n\n").concat(g4.join("\n"), "\n")));
        } else {
          var b4 = x3(l4), d5 = "", R4 = j3[i5];
          "notDeepEqual" === i5 || "notEqual" === i5 ? (b4 = "".concat(j3[i5], "\n\n").concat(b4)).length > 1024 && (b4 = "".concat(b4.slice(0, 1021), "...")) : (d5 = "".concat(x3(f5)), b4.length > 512 && (b4 = "".concat(b4.slice(0, 509), "...")), d5.length > 512 && (d5 = "".concat(d5.slice(0, 509), "...")), "deepEqual" === i5 || "equal" === i5 ? b4 = "".concat(R4, "\n\n").concat(b4, "\n\nshould equal\n\n") : d5 = " ".concat(i5, " ").concat(d5)), r6 = c4(this, h5(e5).call(this, "".concat(b4).concat(d5)));
        }
        return Error.stackTraceLimit = s5, r6.generatedMessage = !o5, Object.defineProperty(a4(r6), "name", { value: "AssertionError [ERR_ASSERTION]", enumerable: false, writable: true, configurable: true }), r6.code = "ERR_ASSERTION", r6.actual = l4, r6.expected = f5, r6.operator = i5, Error.captureStackTrace && Error.captureStackTrace(a4(r6), u5), r6.stack, r6.name = "AssertionError", c4(r6);
      }
      var i4, u4;
      return !function(t6, e6) {
        if ("function" != typeof e6 && null !== e6)
          throw new TypeError("Super expression must either be null or a function");
        t6.prototype = Object.create(e6 && e6.prototype, { constructor: { value: t6, writable: true, configurable: true } }), e6 && g3(t6, e6);
      }(e5, t5), i4 = e5, (u4 = [{ key: "toString", value: function() {
        return "".concat(this.name, " [").concat(this.code, "]: ").concat(this.message);
      } }, { key: b3.custom, value: function(t6, e6) {
        return b3(this, function(t7) {
          for (var e7 = 1; e7 < arguments.length; e7++) {
            var n5 = null != arguments[e7] ? arguments[e7] : {}, o5 = Object.keys(n5);
            "function" == typeof Object.getOwnPropertySymbols && (o5 = o5.concat(Object.getOwnPropertySymbols(n5).filter(function(t8) {
              return Object.getOwnPropertyDescriptor(n5, t8).enumerable;
            }))), o5.forEach(function(e8) {
              r5(t7, e8, n5[e8]);
            });
          }
          return t7;
        }({}, e6, { customInspect: false, depth: 0 }));
      } }]) && o4(i4.prototype, u4), e5;
    }(f4(Error));
    return u$5 = R3;
  }
  function s$3(t5, e5) {
    return function(t6) {
      if (Array.isArray(t6))
        return t6;
    }(t5) || function(t6, e6) {
      var n4 = [], r5 = true, o4 = false, c4 = void 0;
      try {
        for (var a4, i4 = t6[Symbol.iterator](); !(r5 = (a4 = i4.next()).done) && (n4.push(a4.value), !e6 || n4.length !== e6); r5 = true)
          ;
      } catch (t7) {
        o4 = true, c4 = t7;
      } finally {
        try {
          r5 || null == i4.return || i4.return();
        } finally {
          if (o4)
            throw c4;
        }
      }
      return n4;
    }(t5, e5) || function() {
      throw new TypeError("Invalid attempt to destructure non-iterable instance");
    }();
  }
  function p$3(t5) {
    return (p$3 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(t6) {
      return typeof t6;
    } : function(t6) {
      return t6 && "function" == typeof Symbol && t6.constructor === Symbol && t6 !== Symbol.prototype ? "symbol" : typeof t6;
    })(t5);
  }
  var g$1 = void 0 !== /a/g.flags;
  var h$1 = function(t5) {
    var e5 = [];
    return t5.forEach(function(t6) {
      return e5.push(t6);
    }), e5;
  };
  var y$2 = function(t5) {
    var e5 = [];
    return t5.forEach(function(t6, n4) {
      return e5.push([n4, t6]);
    }), e5;
  };
  var b$1 = Object.is ? Object.is : m3;
  var v$1 = Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols : function() {
    return [];
  };
  var d$1 = Number.isNaN ? Number.isNaN : f$5;
  function m$2(t5) {
    return t5.call.bind(t5);
  }
  var E2 = m$2(Object.prototype.hasOwnProperty);
  var w$1 = m$2(Object.prototype.propertyIsEnumerable);
  var S2 = m$2(Object.prototype.toString);
  var j$1 = X.types;
  var O2 = j$1.isAnyArrayBuffer;
  var x2 = j$1.isArrayBufferView;
  var q2 = j$1.isDate;
  var R2 = j$1.isMap;
  var A$1 = j$1.isRegExp;
  var k2 = j$1.isSet;
  var _2 = j$1.isNativeError;
  var T3 = j$1.isBoxedPrimitive;
  var P$1 = j$1.isNumberObject;
  var I2 = j$1.isStringObject;
  var D2 = j$1.isBooleanObject;
  var F2 = j$1.isBigIntObject;
  var N$1 = j$1.isSymbolObject;
  var L2 = j$1.isFloat32Array;
  var M2 = j$1.isFloat64Array;
  function U2(t5) {
    if (0 === t5.length || t5.length > 10)
      return true;
    for (var e5 = 0; e5 < t5.length; e5++) {
      var n4 = t5.charCodeAt(e5);
      if (n4 < 48 || n4 > 57)
        return true;
    }
    return 10 === t5.length && t5 >= Math.pow(2, 32);
  }
  function G2(t5) {
    return Object.keys(t5).filter(U2).concat(v$1(t5).filter(Object.prototype.propertyIsEnumerable.bind(t5)));
  }
  function V2(t5, e5) {
    if (t5 === e5)
      return 0;
    for (var n4 = t5.length, r5 = e5.length, o4 = 0, c4 = Math.min(n4, r5); o4 < c4; ++o4)
      if (t5[o4] !== e5[o4]) {
        n4 = t5[o4], r5 = e5[o4];
        break;
      }
    return n4 < r5 ? -1 : r5 < n4 ? 1 : 0;
  }
  function B2(t5, e5, n4, r5) {
    if (t5 === e5)
      return 0 !== t5 || (!n4 || b$1(t5, e5));
    if (n4) {
      if ("object" !== p$3(t5))
        return "number" == typeof t5 && d$1(t5) && d$1(e5);
      if ("object" !== p$3(e5) || null === t5 || null === e5)
        return false;
      if (Object.getPrototypeOf(t5) !== Object.getPrototypeOf(e5))
        return false;
    } else {
      if (null === t5 || "object" !== p$3(t5))
        return (null === e5 || "object" !== p$3(e5)) && t5 == e5;
      if (null === e5 || "object" !== p$3(e5))
        return false;
    }
    var o4, c4, a4, i4, u4 = S2(t5);
    if (u4 !== S2(e5))
      return false;
    if (Array.isArray(t5)) {
      if (t5.length !== e5.length)
        return false;
      var l4 = G2(t5), f4 = G2(e5);
      return l4.length === f4.length && C2(t5, e5, n4, r5, 1, l4);
    }
    if ("[object Object]" === u4 && (!R2(t5) && R2(e5) || !k2(t5) && k2(e5)))
      return false;
    if (q2(t5)) {
      if (!q2(e5) || Date.prototype.getTime.call(t5) !== Date.prototype.getTime.call(e5))
        return false;
    } else if (A$1(t5)) {
      if (!A$1(e5) || (a4 = t5, i4 = e5, !(g$1 ? a4.source === i4.source && a4.flags === i4.flags : RegExp.prototype.toString.call(a4) === RegExp.prototype.toString.call(i4))))
        return false;
    } else if (_2(t5) || t5 instanceof Error) {
      if (t5.message !== e5.message || t5.name !== e5.name)
        return false;
    } else {
      if (x2(t5)) {
        if (n4 || !L2(t5) && !M2(t5)) {
          if (!function(t6, e6) {
            return t6.byteLength === e6.byteLength && 0 === V2(new Uint8Array(t6.buffer, t6.byteOffset, t6.byteLength), new Uint8Array(e6.buffer, e6.byteOffset, e6.byteLength));
          }(t5, e5))
            return false;
        } else if (!function(t6, e6) {
          if (t6.byteLength !== e6.byteLength)
            return false;
          for (var n5 = 0; n5 < t6.byteLength; n5++)
            if (t6[n5] !== e6[n5])
              return false;
          return true;
        }(t5, e5))
          return false;
        var s4 = G2(t5), h5 = G2(e5);
        return s4.length === h5.length && C2(t5, e5, n4, r5, 0, s4);
      }
      if (k2(t5))
        return !(!k2(e5) || t5.size !== e5.size) && C2(t5, e5, n4, r5, 2);
      if (R2(t5))
        return !(!R2(e5) || t5.size !== e5.size) && C2(t5, e5, n4, r5, 3);
      if (O2(t5)) {
        if (c4 = e5, (o4 = t5).byteLength !== c4.byteLength || 0 !== V2(new Uint8Array(o4), new Uint8Array(c4)))
          return false;
      } else if (T3(t5) && !function(t6, e6) {
        return P$1(t6) ? P$1(e6) && b$1(Number.prototype.valueOf.call(t6), Number.prototype.valueOf.call(e6)) : I2(t6) ? I2(e6) && String.prototype.valueOf.call(t6) === String.prototype.valueOf.call(e6) : D2(t6) ? D2(e6) && Boolean.prototype.valueOf.call(t6) === Boolean.prototype.valueOf.call(e6) : F2(t6) ? F2(e6) && BigInt.prototype.valueOf.call(t6) === BigInt.prototype.valueOf.call(e6) : N$1(e6) && Symbol.prototype.valueOf.call(t6) === Symbol.prototype.valueOf.call(e6);
      }(t5, e5))
        return false;
    }
    return C2(t5, e5, n4, r5, 0);
  }
  function z2(t5, e5) {
    return e5.filter(function(e6) {
      return w$1(t5, e6);
    });
  }
  function C2(t5, e5, n4, r5, o4, c4) {
    if (5 === arguments.length) {
      c4 = Object.keys(t5);
      var a4 = Object.keys(e5);
      if (c4.length !== a4.length)
        return false;
    }
    for (var i4 = 0; i4 < c4.length; i4++)
      if (!E2(e5, c4[i4]))
        return false;
    if (n4 && 5 === arguments.length) {
      var u4 = v$1(t5);
      if (0 !== u4.length) {
        var l4 = 0;
        for (i4 = 0; i4 < u4.length; i4++) {
          var f4 = u4[i4];
          if (w$1(t5, f4)) {
            if (!w$1(e5, f4))
              return false;
            c4.push(f4), l4++;
          } else if (w$1(e5, f4))
            return false;
        }
        var s4 = v$1(e5);
        if (u4.length !== s4.length && z2(e5, s4).length !== l4)
          return false;
      } else {
        var p4 = v$1(e5);
        if (0 !== p4.length && 0 !== z2(e5, p4).length)
          return false;
      }
    }
    if (0 === c4.length && (0 === o4 || 1 === o4 && 0 === t5.length || 0 === t5.size))
      return true;
    if (void 0 === r5)
      r5 = { val1: /* @__PURE__ */ new Map(), val2: /* @__PURE__ */ new Map(), position: 0 };
    else {
      var g3 = r5.val1.get(t5);
      if (void 0 !== g3) {
        var h5 = r5.val2.get(e5);
        if (void 0 !== h5)
          return g3 === h5;
      }
      r5.position++;
    }
    r5.val1.set(t5, r5.position), r5.val2.set(e5, r5.position);
    var y4 = Q2(t5, e5, n4, c4, r5, o4);
    return r5.val1.delete(t5), r5.val2.delete(e5), y4;
  }
  function Y2(t5, e5, n4, r5) {
    for (var o4 = h$1(t5), c4 = 0; c4 < o4.length; c4++) {
      var a4 = o4[c4];
      if (B2(e5, a4, n4, r5))
        return t5.delete(a4), true;
    }
    return false;
  }
  function W2(t5) {
    switch (p$3(t5)) {
      case "undefined":
        return null;
      case "object":
        return;
      case "symbol":
        return false;
      case "string":
        t5 = +t5;
      case "number":
        if (d$1(t5))
          return false;
    }
    return true;
  }
  function H2(t5, e5, n4) {
    var r5 = W2(n4);
    return null != r5 ? r5 : e5.has(r5) && !t5.has(r5);
  }
  function J2(t5, e5, n4, r5, o4) {
    var c4 = W2(n4);
    if (null != c4)
      return c4;
    var a4 = e5.get(c4);
    return !(void 0 === a4 && !e5.has(c4) || !B2(r5, a4, false, o4)) && (!t5.has(c4) && B2(r5, a4, false, o4));
  }
  function K2(t5, e5, n4, r5, o4, c4) {
    for (var a4 = h$1(t5), i4 = 0; i4 < a4.length; i4++) {
      var u4 = a4[i4];
      if (B2(n4, u4, o4, c4) && B2(r5, e5.get(u4), o4, c4))
        return t5.delete(u4), true;
    }
    return false;
  }
  function Q2(t5, e5, n4, r5, o4, c4) {
    var a4 = 0;
    if (2 === c4) {
      if (!function(t6, e6, n5, r6) {
        for (var o5 = null, c5 = h$1(t6), a5 = 0; a5 < c5.length; a5++) {
          var i5 = c5[a5];
          if ("object" === p$3(i5) && null !== i5)
            null === o5 && (o5 = /* @__PURE__ */ new Set()), o5.add(i5);
          else if (!e6.has(i5)) {
            if (n5)
              return false;
            if (!H2(t6, e6, i5))
              return false;
            null === o5 && (o5 = /* @__PURE__ */ new Set()), o5.add(i5);
          }
        }
        if (null !== o5) {
          for (var u5 = h$1(e6), l5 = 0; l5 < u5.length; l5++) {
            var f4 = u5[l5];
            if ("object" === p$3(f4) && null !== f4) {
              if (!Y2(o5, f4, n5, r6))
                return false;
            } else if (!n5 && !t6.has(f4) && !Y2(o5, f4, n5, r6))
              return false;
          }
          return 0 === o5.size;
        }
        return true;
      }(t5, e5, n4, o4))
        return false;
    } else if (3 === c4) {
      if (!function(t6, e6, n5, r6) {
        for (var o5 = null, c5 = y$2(t6), a5 = 0; a5 < c5.length; a5++) {
          var i5 = s$3(c5[a5], 2), u5 = i5[0], l5 = i5[1];
          if ("object" === p$3(u5) && null !== u5)
            null === o5 && (o5 = /* @__PURE__ */ new Set()), o5.add(u5);
          else {
            var f4 = e6.get(u5);
            if (void 0 === f4 && !e6.has(u5) || !B2(l5, f4, n5, r6)) {
              if (n5)
                return false;
              if (!J2(t6, e6, u5, l5, r6))
                return false;
              null === o5 && (o5 = /* @__PURE__ */ new Set()), o5.add(u5);
            }
          }
        }
        if (null !== o5) {
          for (var g3 = y$2(e6), h5 = 0; h5 < g3.length; h5++) {
            var b3 = s$3(g3[h5], 2), v4 = (u5 = b3[0], b3[1]);
            if ("object" === p$3(u5) && null !== u5) {
              if (!K2(o5, t6, u5, v4, n5, r6))
                return false;
            } else if (!(n5 || t6.has(u5) && B2(t6.get(u5), v4, false, r6) || K2(o5, t6, u5, v4, false, r6)))
              return false;
          }
          return 0 === o5.size;
        }
        return true;
      }(t5, e5, n4, o4))
        return false;
    } else if (1 === c4)
      for (; a4 < t5.length; a4++) {
        if (!E2(t5, a4)) {
          if (E2(e5, a4))
            return false;
          for (var i4 = Object.keys(t5); a4 < i4.length; a4++) {
            var u4 = i4[a4];
            if (!E2(e5, u4) || !B2(t5[u4], e5[u4], n4, o4))
              return false;
          }
          return i4.length === Object.keys(e5).length;
        }
        if (!E2(e5, a4) || !B2(t5[a4], e5[a4], n4, o4))
          return false;
      }
    for (a4 = 0; a4 < r5.length; a4++) {
      var l4 = r5[a4];
      if (!B2(t5[l4], e5[l4], n4, o4))
        return false;
    }
    return true;
  }
  var X2 = { isDeepEqual: function(t5, e5) {
    return B2(t5, e5, false);
  }, isDeepStrictEqual: function(t5, e5) {
    return B2(t5, e5, true);
  } };
  var Z2 = {};
  var $$1 = false;
  function tt() {
    if ($$1)
      return Z2;
    $$1 = true;
    var o4 = T;
    function c4(t5) {
      return (c4 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(t6) {
        return typeof t6;
      } : function(t6) {
        return t6 && "function" == typeof Symbol && t6.constructor === Symbol && t6 !== Symbol.prototype ? "symbol" : typeof t6;
      })(t5);
    }
    var a4, u4, l4 = i$5().codes, s4 = l4.ERR_AMBIGUOUS_ARGUMENT, p4 = l4.ERR_INVALID_ARG_TYPE, g3 = l4.ERR_INVALID_ARG_VALUE, h5 = l4.ERR_INVALID_RETURN_VALUE, y4 = l4.ERR_MISSING_ARGS, b3 = f$6(), v4 = X.inspect, d4 = X.types, m$12 = d4.isPromise, E3 = d4.isRegExp, w3 = Object.assign ? Object.assign : r4.assign, S3 = Object.is ? Object.is : m3;
    function j3() {
      a4 = X2.isDeepEqual, u4 = X2.isDeepStrictEqual;
    }
    var O3 = false, x3 = Z2 = k3, q3 = {};
    function R3(t5) {
      if (t5.message instanceof Error)
        throw t5.message;
      throw new b3(t5);
    }
    function A3(t5, e5, n4, r5) {
      if (!n4) {
        var o5 = false;
        if (0 === e5)
          o5 = true, r5 = "No value argument passed to `assert.ok()`";
        else if (r5 instanceof Error)
          throw r5;
        var c5 = new b3({ actual: n4, expected: true, message: r5, operator: "==", stackStartFn: t5 });
        throw c5.generatedMessage = o5, c5;
      }
    }
    function k3() {
      for (var t5 = arguments.length, e5 = new Array(t5), n4 = 0; n4 < t5; n4++)
        e5[n4] = arguments[n4];
      A3.apply(void 0, [k3, e5.length].concat(e5));
    }
    x3.fail = function t5(e5, n4, r5, c5, a5) {
      var i4, u5 = arguments.length;
      if (0 === u5)
        i4 = "Failed";
      else if (1 === u5)
        r5 = e5, e5 = void 0;
      else {
        if (false === O3) {
          O3 = true;
          var l5 = o4.emitWarning ? o4.emitWarning : console.warn.bind(console);
          l5("assert.fail() with more than one argument is deprecated. Please use assert.strictEqual() instead or only pass a message.", "DeprecationWarning", "DEP0094");
        }
        2 === u5 && (c5 = "!=");
      }
      if (r5 instanceof Error)
        throw r5;
      var f4 = { actual: e5, expected: n4, operator: void 0 === c5 ? "fail" : c5, stackStartFn: a5 || t5 };
      void 0 !== r5 && (f4.message = r5);
      var s5 = new b3(f4);
      throw i4 && (s5.message = i4, s5.generatedMessage = true), s5;
    }, x3.AssertionError = b3, x3.ok = k3, x3.equal = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      e5 != n4 && R3({ actual: e5, expected: n4, message: r5, operator: "==", stackStartFn: t5 });
    }, x3.notEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      e5 == n4 && R3({ actual: e5, expected: n4, message: r5, operator: "!=", stackStartFn: t5 });
    }, x3.deepEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      void 0 === a4 && j3(), a4(e5, n4) || R3({ actual: e5, expected: n4, message: r5, operator: "deepEqual", stackStartFn: t5 });
    }, x3.notDeepEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      void 0 === a4 && j3(), a4(e5, n4) && R3({ actual: e5, expected: n4, message: r5, operator: "notDeepEqual", stackStartFn: t5 });
    }, x3.deepStrictEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      void 0 === a4 && j3(), u4(e5, n4) || R3({ actual: e5, expected: n4, message: r5, operator: "deepStrictEqual", stackStartFn: t5 });
    }, x3.notDeepStrictEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      void 0 === a4 && j3();
      u4(e5, n4) && R3({ actual: e5, expected: n4, message: r5, operator: "notDeepStrictEqual", stackStartFn: t5 });
    }, x3.strictEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      S3(e5, n4) || R3({ actual: e5, expected: n4, message: r5, operator: "strictEqual", stackStartFn: t5 });
    }, x3.notStrictEqual = function t5(e5, n4, r5) {
      if (arguments.length < 2)
        throw new y4("actual", "expected");
      S3(e5, n4) && R3({ actual: e5, expected: n4, message: r5, operator: "notStrictEqual", stackStartFn: t5 });
    };
    var _3 = function t5(e5, n4, r5) {
      var o5 = this;
      !function(t6, e6) {
        if (!(t6 instanceof e6))
          throw new TypeError("Cannot call a class as a function");
      }(this, t5), n4.forEach(function(t6) {
        t6 in e5 && (void 0 !== r5 && "string" == typeof r5[t6] && E3(e5[t6]) && e5[t6].test(r5[t6]) ? o5[t6] = r5[t6] : o5[t6] = e5[t6]);
      });
    };
    function T4(t5, e5, n4, r5, o5, c5) {
      if (!(n4 in t5) || !u4(t5[n4], e5[n4])) {
        if (!r5) {
          var a5 = new _3(t5, o5), i4 = new _3(e5, o5, t5), l5 = new b3({ actual: a5, expected: i4, operator: "deepStrictEqual", stackStartFn: c5 });
          throw l5.actual = t5, l5.expected = e5, l5.operator = c5.name, l5;
        }
        R3({ actual: t5, expected: e5, message: r5, operator: c5.name, stackStartFn: c5 });
      }
    }
    function P3(t5, e5, n4, r5) {
      if ("function" != typeof e5) {
        if (E3(e5))
          return e5.test(t5);
        if (2 === arguments.length)
          throw new p4("expected", ["Function", "RegExp"], e5);
        if ("object" !== c4(t5) || null === t5) {
          var o5 = new b3({ actual: t5, expected: e5, message: n4, operator: "deepStrictEqual", stackStartFn: r5 });
          throw o5.operator = r5.name, o5;
        }
        var i4 = Object.keys(e5);
        if (e5 instanceof Error)
          i4.push("name", "message");
        else if (0 === i4.length)
          throw new g3("error", e5, "may not be an empty object");
        return void 0 === a4 && j3(), i4.forEach(function(o6) {
          "string" == typeof t5[o6] && E3(e5[o6]) && e5[o6].test(t5[o6]) || T4(t5, e5, o6, n4, i4, r5);
        }), true;
      }
      return void 0 !== e5.prototype && t5 instanceof e5 || !Error.isPrototypeOf(e5) && true === e5.call({}, t5);
    }
    function I3(t5) {
      if ("function" != typeof t5)
        throw new p4("fn", "Function", t5);
      try {
        t5();
      } catch (t6) {
        return t6;
      }
      return q3;
    }
    function D3(t5) {
      return m$12(t5) || null !== t5 && "object" === c4(t5) && "function" == typeof t5.then && "function" == typeof t5.catch;
    }
    function F3(t5) {
      return Promise.resolve().then(function() {
        var e5;
        if ("function" == typeof t5) {
          if (!D3(e5 = t5()))
            throw new h5("instance of Promise", "promiseFn", e5);
        } else {
          if (!D3(t5))
            throw new p4("promiseFn", ["Function", "Promise"], t5);
          e5 = t5;
        }
        return Promise.resolve().then(function() {
          return e5;
        }).then(function() {
          return q3;
        }).catch(function(t6) {
          return t6;
        });
      });
    }
    function N3(t5, e5, n4, r5) {
      if ("string" == typeof n4) {
        if (4 === arguments.length)
          throw new p4("error", ["Object", "Error", "Function", "RegExp"], n4);
        if ("object" === c4(e5) && null !== e5) {
          if (e5.message === n4)
            throw new s4("error/message", 'The error message "'.concat(e5.message, '" is identical to the message.'));
        } else if (e5 === n4)
          throw new s4("error/message", 'The error "'.concat(e5, '" is identical to the message.'));
        r5 = n4, n4 = void 0;
      } else if (null != n4 && "object" !== c4(n4) && "function" != typeof n4)
        throw new p4("error", ["Object", "Error", "Function", "RegExp"], n4);
      if (e5 === q3) {
        var o5 = "";
        n4 && n4.name && (o5 += " (".concat(n4.name, ")")), o5 += r5 ? ": ".concat(r5) : ".";
        var a5 = "rejects" === t5.name ? "rejection" : "exception";
        R3({ actual: void 0, expected: n4, operator: t5.name, message: "Missing expected ".concat(a5).concat(o5), stackStartFn: t5 });
      }
      if (n4 && !P3(e5, n4, r5, t5))
        throw e5;
    }
    function L3(t5, e5, n4, r5) {
      if (e5 !== q3) {
        if ("string" == typeof n4 && (r5 = n4, n4 = void 0), !n4 || P3(e5, n4)) {
          var o5 = r5 ? ": ".concat(r5) : ".", c5 = "doesNotReject" === t5.name ? "rejection" : "exception";
          R3({ actual: e5, expected: n4, operator: t5.name, message: "Got unwanted ".concat(c5).concat(o5, "\n") + 'Actual message: "'.concat(e5 && e5.message, '"'), stackStartFn: t5 });
        }
        throw e5;
      }
    }
    function M3() {
      for (var t5 = arguments.length, e5 = new Array(t5), n4 = 0; n4 < t5; n4++)
        e5[n4] = arguments[n4];
      A3.apply(void 0, [M3, e5.length].concat(e5));
    }
    return x3.throws = function t5(e5) {
      for (var n4 = arguments.length, r5 = new Array(n4 > 1 ? n4 - 1 : 0), o5 = 1; o5 < n4; o5++)
        r5[o5 - 1] = arguments[o5];
      N3.apply(void 0, [t5, I3(e5)].concat(r5));
    }, x3.rejects = function t5(e5) {
      for (var n4 = arguments.length, r5 = new Array(n4 > 1 ? n4 - 1 : 0), o5 = 1; o5 < n4; o5++)
        r5[o5 - 1] = arguments[o5];
      return F3(e5).then(function(e6) {
        return N3.apply(void 0, [t5, e6].concat(r5));
      });
    }, x3.doesNotThrow = function t5(e5) {
      for (var n4 = arguments.length, r5 = new Array(n4 > 1 ? n4 - 1 : 0), o5 = 1; o5 < n4; o5++)
        r5[o5 - 1] = arguments[o5];
      L3.apply(void 0, [t5, I3(e5)].concat(r5));
    }, x3.doesNotReject = function t5(e5) {
      for (var n4 = arguments.length, r5 = new Array(n4 > 1 ? n4 - 1 : 0), o5 = 1; o5 < n4; o5++)
        r5[o5 - 1] = arguments[o5];
      return F3(e5).then(function(e6) {
        return L3.apply(void 0, [t5, e6].concat(r5));
      });
    }, x3.ifError = function t5(e5) {
      if (null != e5) {
        var n4 = "ifError got unwanted exception: ";
        "object" === c4(e5) && "string" == typeof e5.message ? 0 === e5.message.length && e5.constructor ? n4 += e5.constructor.name : n4 += e5.message : n4 += v4(e5);
        var r5 = new b3({ actual: e5, expected: null, operator: "ifError", message: n4, stackStartFn: t5 }), o5 = e5.stack;
        if ("string" == typeof o5) {
          var a5 = o5.split("\n");
          a5.shift();
          for (var i4 = r5.stack.split("\n"), u5 = 0; u5 < a5.length; u5++) {
            var l5 = i4.indexOf(a5[u5]);
            if (-1 !== l5) {
              i4 = i4.slice(0, l5);
              break;
            }
          }
          r5.stack = "".concat(i4.join("\n"), "\n").concat(a5.join("\n"));
        }
        throw r5;
      }
    }, x3.strict = w3(M3, x3, { equal: x3.strictEqual, deepEqual: x3.deepStrictEqual, notEqual: x3.notStrictEqual, notDeepEqual: x3.notDeepStrictEqual }), x3.strict.strict = x3.strict, Z2;
  }
  var et = tt();
  et.AssertionError;
  et.deepEqual;
  et.deepStrictEqual;
  et.doesNotReject;
  et.doesNotThrow;
  et.equal;
  et.fail;
  et.ifError;
  et.notDeepEqual;
  et.notDeepStrictEqual;
  et.notEqual;
  et.notStrictEqual;
  et.ok;
  et.rejects;
  et.strict;
  et.strictEqual;
  et.throws;
  et.AssertionError;
  et.deepEqual;
  et.deepStrictEqual;
  et.doesNotReject;
  et.doesNotThrow;
  et.equal;
  et.fail;
  et.ifError;
  et.notDeepEqual;
  et.notDeepStrictEqual;
  et.notEqual;
  et.notStrictEqual;
  et.ok;
  et.rejects;
  et.strict;
  et.strictEqual;
  et.throws;
  var AssertionError = et.AssertionError;
  var deepEqual = et.deepEqual;
  var deepStrictEqual = et.deepStrictEqual;
  var doesNotReject = et.doesNotReject;
  var doesNotThrow = et.doesNotThrow;
  var equal = et.equal;
  var fail = et.fail;
  var ifError = et.ifError;
  var notDeepEqual = et.notDeepEqual;
  var notDeepStrictEqual = et.notDeepStrictEqual;
  var notEqual = et.notEqual;
  var notStrictEqual = et.notStrictEqual;
  var ok = et.ok;
  var rejects = et.rejects;
  var strict = et.strict;
  var strictEqual = et.strictEqual;
  var throws = et.throws;

  // src/utils.ts
  init_global();
  init_dirname();
  init_filename();
  init_buffer2();
  init_process2();
  function sleep(milliseconds) {
    return new Promise((r5) => setTimeout(r5, milliseconds));
  }

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
    constructor(config) {
      this.config = config;
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
    const process = (key, value) => {
      if (isDefined(value)) {
        if (Array.isArray(value)) {
          value.forEach((v4) => {
            process(key, v4);
          });
        } else if (typeof value === "object") {
          Object.entries(value).forEach(([k3, v4]) => {
            process(`${key}[${k3}]`, v4);
          });
        } else {
          append2(key, value);
        }
      }
    };
    Object.entries(params).forEach(([key, value]) => {
      process(key, value);
    });
    if (qs.length > 0) {
      return `?${qs.join("&")}`;
    }
    return "";
  };
  var getUrl = (config, options) => {
    const encoder = config.ENCODE_PATH || encodeURI;
    const path = options.url.replace("{api-version}", config.VERSION).replace(/{(.*?)}/g, (substring, group) => {
      if (options.path?.hasOwnProperty(group)) {
        return encoder(String(options.path[group]));
      }
      return substring;
    });
    const url = `${config.BASE}${path}`;
    if (options.query) {
      return `${url}${getQueryString(options.query)}`;
    }
    return url;
  };
  var getFormData = (options) => {
    if (options.formData) {
      const formData = new import_form_data.default();
      const process = (key, value) => {
        if (isString2(value) || isBlob2(value)) {
          formData.append(key, value);
        } else {
          formData.append(key, JSON.stringify(value));
        }
      };
      Object.entries(options.formData).filter(([_3, value]) => isDefined(value)).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          value.forEach((v4) => process(key, v4));
        } else {
          process(key, value);
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
  var getHeaders = async (config, options, formData) => {
    const token = await resolve(options, config.TOKEN);
    const username = await resolve(options, config.USERNAME);
    const password = await resolve(options, config.PASSWORD);
    const additionalHeaders = await resolve(options, config.HEADERS);
    const formHeaders = typeof formData?.getHeaders === "function" && formData?.getHeaders() || {};
    const headers = Object.entries({
      Accept: "application/json",
      ...additionalHeaders,
      ...options.headers,
      ...formHeaders
    }).filter(([_3, value]) => isDefined(value)).reduce((headers2, [key, value]) => ({
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
  var sendRequest = async (config, options, url, body, formData, headers, onCancel) => {
    const source = axios_default.CancelToken.source();
    const requestConfig = {
      url,
      headers,
      data: body ?? formData,
      method: options.method,
      withCredentials: config.WITH_CREDENTIALS,
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
  var request = (config, options) => {
    return new CancelablePromise(async (resolve2, reject, onCancel) => {
      try {
        const url = getUrl(config, options);
        const formData = getFormData(options);
        const body = getRequestBody(options);
        const headers = await getHeaders(config, options, formData);
        if (!onCancel.isCancelled) {
          const response = await sendRequest(config, options, url, body, formData, headers, onCancel);
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
    constructor(config) {
      super(config);
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
    constructor(config, HttpRequest = AxiosHttpRequest) {
      this.request = new HttpRequest({
        BASE: config?.BASE ?? "",
        VERSION: config?.VERSION ?? "0.1.0",
        WITH_CREDENTIALS: config?.WITH_CREDENTIALS ?? false,
        CREDENTIALS: config?.CREDENTIALS ?? "include",
        TOKEN: config?.TOKEN,
        USERNAME: config?.USERNAME,
        PASSWORD: config?.PASSWORD,
        HEADERS: config?.HEADERS,
        ENCODE_PATH: config?.ENCODE_PATH
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

  // src/TabbyAgent.ts
  var TabbyAgent = class extends EventEmitter {
    constructor() {
      super();
      this.serverUrl = "http://127.0.0.1:5000";
      this.status = "connecting";
      this.ping();
      this.api = new TabbyApi({ BASE: this.serverUrl });
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
        const response = await axios_default.get(`${this.serverUrl}/`);
        strict(response.status == 200);
        this.changeStatus("ready");
        return true;
      } catch (e5) {
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
      return new CancelablePromise((resolve2, reject, onCancel) => {
        promise.then((resp) => {
          this.changeStatus("ready");
          resolve2(resp);
        }).catch((err) => {
          reject(err);
        }).catch((err) => {
          this.changeStatus("disconnected");
          reject(err);
        }).catch((err) => {
          reject(err);
        });
        onCancel(() => {
          promise.cancel();
        });
      });
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
      const promise = this.api.default.completionsV1CompletionsPost(request2);
      return this.wrapApiPromise(promise);
    }
    postEvent(request2) {
      const promise = this.api.default.eventsV1EventsPost(request2);
      return this.wrapApiPromise(promise);
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

@jspm/core/nodelibs/browser/assert.js:
  (*!
   * The buffer module from node.js, for the browser.
   *
   * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
   * @license  MIT
   *)
*/
//# sourceMappingURL=index.global.js.map

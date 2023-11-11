const arr = ["a", "b", "c"];

function myEncode(arr) {
  return arr
    .map(item => item.toLocaleUpperCase())
    .map(item => `(${item})`)
    .map(item => Buffer.from(item).toString("base64"));
}
console.log(myEncode(arr));

function myDecode(arr) {
  return arr
    .map⏩⏭(item => Buffer.from(item, "base64").toString("ascii"))
    .map(item => item.toLocaleUpperCase());
}⏮}⏪
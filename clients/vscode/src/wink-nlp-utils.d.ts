declare module "wink-nlp-utils" {
  namespace winkUtils {
    // REF: https://winkjs.org/wink-nlp-utils/string.html
    namespace string {
      function tokenize0(s: string): string[];
      function stem(word: string): string;
    }
    // REF: https://winkjs.org/wink-nlp-utils/tokens.html
    namespace tokens {
      function removeWords(words: string[]): string[];
      function stem(words: string[]): string[];
    }
  }

  export default winkUtils;
}

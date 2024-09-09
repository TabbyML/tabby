# An Introduction to Codelens

## Preface

When writing code, you may wonder how to run it in the terminal, especially if you are fresh. Nowadays, many tools and plugins integrate extensions to solve this problem. For example, if you are learning write rust code on [vscode](https://code.visualstudio.com/), only install an extension [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer). After writing rust test code, you could see the Run Test or Debug button above the code, like the following picture:

![image.png](./image.png)

Click one of the buttons, then it will run the test or debug code directly. You don’t need to worry about the test command or how to configure the debug environment. And this feature is based on CodeLens.

## What is Codelens

CodeLens is a useful feature provided by many modern IDEs, such as Visual Studio and IntelliJ IDEA. It's designed to improve developers productivity and enhance the development experience. CodeLens aims to offer meta information about the code or execute actions directly within the IDE. Specifically, CodeLens can perform the following tasks without modifying the codebase:

- The counts of function references;
- Git history information, such as who changed the code and when;
- Code Action Executor, providing a button above the code that you can click to execute the code;
- Code analysis, analyze the complexity of code;

Without these features, developers often need to switch their windows between terminal and IDE, which might disrupt the workflow.

## How to combine it with Tabby

In Tabby, there are two features that might be able to implement based on CodeLens. The first feature would help developer to understand large codebase, especially in projects within millions of lines of codes. For instance, Tabby could provide a guideline to locate the entry function and which files or function they have used in the entry function. On the other side, it could provide information about the function purpose, references and its complexity. Going further, it can provide a better version about the function if complexity is too high or the code may has potential bugs.

The second is supporting different language test suite tools internally. When generating the code, Tabby could also generate the test code at the same time. After that, developer could run the code directly, this feature could let the developers spot on their key business logic **codes** instead of some utils functions like the following code:

```rust
fn find_max_ele_in_vec(v: &Vec<i64>) -> Option<i64> {
  if v.is_empty() {
    return None;
  }

  let mut max = v.first().unwrap();
  for item in v {
    if max < item {
      max = item;
    }
  }

  Some(*max)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_find_max_ele_in_vec() {
    let case: Vec<i64> = vec![];
    let res = find_max_ele_in_vec(&case);
    assert!(res.is_none);
  }
}
```

and then developers don’t need to care about there are bugs in the generated code, the test code could ensure the code equality in some degree.

These two feature are focus on developer experience, and make every developer more productive in their project. And I think this is every developer tool care about.

## Conclusion

CodeLens is an enhancement by providing valuable insights and actions directly within the IDE. By integrating these features such as test execution, code analysis and reference tracking, CodeLens empowers developers to enhance the overall coding experience and boost productivity.

import React from "react"
import ReactDOM from "react-dom"
import { StreamlitProvider } from "streamlit-component-lib-react-hooks"
import MyComponent from "./MyComponent"

ReactDOM.render(
  <React.StrictMode>
    <StreamlitProvider>
      <MyComponent />
    </StreamlitProvider>
  </React.StrictMode>,
  document.getElementById("root")
)

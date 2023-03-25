import React from "react"
import ReactDOM from "react-dom"
import { StreamlitProvider } from "streamlit-component-lib-react-hooks"
import Tabby from "./Tabby"

ReactDOM.render(
  <React.StrictMode>
    <StreamlitProvider>
      <Tabby />
    </StreamlitProvider>
  </React.StrictMode>,
  document.getElementById("root")
)

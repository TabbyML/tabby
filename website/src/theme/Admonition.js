import React from 'react';
import Admonition from '@theme-original/Admonition';


export default function AdmonitionWrapper(props) {
  if (props.type === 'subscription') {
    return <Admonition title='SUBSCRIPTION' icon={<span className='text-2xl'>💰</span>}
    >
      {props.children}
    </Admonition>
  } else {
    return <Admonition {...props} />;
  }
}
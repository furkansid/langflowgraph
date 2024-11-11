import { AiChat, useAsStreamAdapter } from '@nlux/react';
import { sendAdapter } from './sendAdapter';
// import { personas } from './personas';
import '@nlux/themes/nova.css';

function LangGraphChat() {
  const adapter = useAsStreamAdapter(sendAdapter, []);

  return (
    <>
  {/* <iframe height="700px" allow="camera; microphone; fullscreen" src="https://trulience.com/avatar/3091004867099640994?dialPageBackground=black&token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJUb2tlbiBmcm9tIGN1c3RvbSBzdHJpbmciLCJleHAiOjQ4NzU0MDAzNTV9.Wmlu3UDqln5wzHiYTxbloG1FpABIjBFVPIICkyru0v3bAh6ty597topbCgIcwlFhUacLFEqEr8BHadu76WSqhw&screenAspectRatio=9:16&controlButtonPosition=center&hideChatInput=true" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen width={1000}></iframe> */}

      {/* <div className="card" style={{minWidth: 1000, maxHeight:200, overflow:"scroll", color:"red", display:"block"}} > */}
        <AiChat
            adapter={adapter}
            // personaOptions={personas}
        />
      {/* </div> */}
    </>
  )
}

export default LangGraphChat;
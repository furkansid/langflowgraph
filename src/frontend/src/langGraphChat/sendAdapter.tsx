import { StreamSend, StreamingAdapterObserver } from '@nlux/react';

// A demo API by NLUX that connects to OpenAI
// and returns a stream of Server-Sent events
// const demoProxyServerUrl = 'https://demo.api.nlux.ai/openai/chat/stream';
const demoProxyServerUrl = 'http://localhost:3000/api/v1/langgraph/bfa69af6-0e35-414e-b401-f8007dab3d1f/stream/';

// Function to send query to the server and receive a stream of chunks as response
export const sendAdapter: StreamSend = async (
    prompt: string,
    observer: StreamingAdapterObserver,
) => {
    const body = {"input": prompt};
    console.log("Flag 1", prompt, body)
    const response = await fetch(demoProxyServerUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });

    if (response.status !== 200) {
        observer.error(new Error('Failed to connect to the server'));
        return;
    }

    if (!response.body) {
        return;
    }

    // Read a stream of server-sent events
    // and feed them to the observer as they are being generated
    const reader = response.body.getReader();
    const textDecoder = new TextDecoder();

    while (true) {
        const {value, done} = await reader.read();
        if (done) {
            break;
        }

        const content = textDecoder.decode(value);
        if (content) {
            observer.next(content);
        }
    }

    observer.complete();
};
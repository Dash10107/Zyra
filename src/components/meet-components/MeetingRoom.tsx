'use client';

import { useState, useEffect, useRef } from 'react';
import {
  CallControls,
  CallParticipantsList,
  CallStatsButton,
  CallingState,
  PaginatedGridLayout,
  SpeakerLayout,
  useCallStateHooks,
} from '@stream-io/video-react-sdk';
import { useRouter, useSearchParams } from 'next/navigation';
import { Users, LayoutList, MessageCircle, Cross } from 'lucide-react';

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import Loader from './Loader';
import EndCallButton from './EndCallButton';
import { cn } from '@/lib/utils';
import axios from 'axios';

type CallLayoutType = 'grid' | 'speaker-left' | 'speaker-right';

const MeetingRoom = () => {
  const searchParams = useSearchParams();
  const isPersonalRoom = !!searchParams.get('personal');
  const router = useRouter();
  const [layout, setLayout] = useState<CallLayoutType>('speaker-left');
  const [showParticipants, setShowParticipants] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showChat, setShowChat] = useState(true);
  const { useCallCallingState } = useCallStateHooks();
  const audioChunks = useRef<Blob[]>([]);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const recordingInterval = useRef<NodeJS.Timeout | null>(null);

  const [participants, setParticipants] = useState<string[]>([]); // Track participants


  // for chat
  const [query, setQuery] = useState<string>("");
  const [queryRes, setQueryRes] = useState<string[]>([]);

  const [chatio, setChatIo] = useState<string>("");

  const userId = "user123"; // Replace with actual user ID
  const userToken = "userToken123"; // Replace with the actual user token

  const [chatRes, setChatRes] = useState<string[]>([]);

  const callingState = useCallCallingState();


  useEffect(() => {
    if (callingState === CallingState.JOINED) {``
      startAudioRecording();
    }

    return () => {
      stopAudioRecording();
    };
  }, [callingState]);

  const startAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      console.log("audio chunk", audioChunks);

      mediaRecorder.current.start();

      recordingInterval.current = setInterval(() => {
        if (mediaRecorder.current?.state === 'recording') {
          mediaRecorder.current.requestData();
          saveAndSendAudio();
        }
      }, 10000); // Capture audio every 3 seconds
    } catch (error) {
      console.error('Error starting audio recording:', error);
    }
  };

  const stopAudioRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
    }
    if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
    }
  };

  const bufferToWave = (abuffer: AudioBuffer, offset = 0): Blob => {  
    const numOfChannels = abuffer.numberOfChannels;  
    const length = abuffer.length * numOfChannels * 2 + 44; // 16-bit PCM  
    const buffer = new ArrayBuffer(length);  
    const view = new DataView(buffer);  
    const channels: Float32Array[] = [];  // Store channel data with the correct type
  
    // Collect the channel data
    for (let i = 0; i < numOfChannels; i++) {  
      channels.push(abuffer.getChannelData(i));  
    }  
  
    // Write WAV header  
    setString(view, 0, 'RIFF');  
    view.setUint32(4, length - 8, true);  
    setString(view, 8, 'WAVE');  
    setString(view, 12, 'fmt ');  
    view.setUint32(16, 16, true); // SubChunk1Size for PCM  
    view.setUint16(20, 1, true); // AudioFormat  
    view.setUint16(22, numOfChannels, true);  
    view.setUint32(24, abuffer.sampleRate, true);  
    view.setUint32(28, abuffer.sampleRate * 2, true); // ByteRate  
    view.setUint16(32, numOfChannels * 2, true); // BlockAlign  
    view.setUint16(34, 16, true); // BitsPerSample  
    setString(view, 36, 'data');  
    view.setUint32(40, length - view.byteLength, true);  
  
    // Write PCM samples  
    let offset1 = 44;  
    for (let i = 0; i < abuffer.length; i++) {  
      for (let channel = 0; channel < numOfChannels; channel++) {  
        const sample = Math.max(-1, Math.min(1, channels[channel][i]));  
        view.setInt16(offset1, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);  
        offset1 += 2;  
      }  
    }  
  
    return new Blob([view], { type: 'audio/wav' });  
  };  
  
  const setString = (dataView: DataView, offset: number, string: string): void => {
    for (let i = 0; i < string.length; i++) {
      dataView.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  

  // const saveAndSendAudio = async () => {
  //   if (audioChunks.current.length > 0) {
  //     try {
  //       const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
  //       audioChunks.current = []; // Clear the chunks for the next recording segment

  //       const formData = new FormData();
  //       formData.append('audio', audioBlob, `audio_${Date.now()}.webm`);
  //       formData.append('language', 'en');

  //       const response = await fetch('https://your-api-endpoint.com/process-audio', {
  //         method: 'POST',
  //         body: formData,
  //       });

  //       if (response.ok) {
  //         const result = await response.json();
  //         console.log('API response:', result);
  //       } else {
  //         console.error('Failed to upload audio:', response.status, response.statusText);
  //       }
  //     } catch (error) {
  //       console.error('Error uploading audio:', error);
  //     }
  //   } else {
  //     console.warn('No audio chunks available to send.');
  //   }
  // };

// FIXME:

const downloadWavBlob = (wavBlob: Blob, fileName: string): void => {
  const url = URL.createObjectURL(wavBlob);  // Create an object URL for the WAV Blob
  const a = document.createElement('a');  // Create an <a> element
  a.href = url;  // Set the href to the Blob URL
  a.download = fileName;  // Set the desired file name
  document.body.appendChild(a);  // Append the <a> element to the DOM
  a.click();  // Programmatically trigger the click to start the download
  document.body.removeChild(a);  // Clean up the DOM by removing the <a> element
  URL.revokeObjectURL(url);  // Release the Blob URL to free memory
};


// FIXME:
const saveAndSendAudio = async () => {
  if (audioChunks.current.length > 0) {
    try {
      // Create an AudioContext to decode the audio data
      const audioContext = new (window.AudioContext)();
      
      // Create a Blob from the audio chunks (webm format)
      const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });

      // Convert the Blob to an ArrayBuffer and decode the audio data
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      console.log('Audio buffer:', audioBuffer);

      // Convert the AudioBuffer to WAV format
      const wavBlob = bufferToWave(audioBuffer);

      // Optionally, trigger the download of the WAV file
      // FIXME: TEMP SHUTDOWN
      // downloadWavBlob(wavBlob, 'audio_output.wav');
      console.log('WAV Blob created:', wavBlob);

      // Reset the chunks for the next recording segment
      audioChunks.current = [];

      // Prepare the FormData to send the WAV file
      const formData = new FormData();
      formData.append('audio', wavBlob, `audio_${Date.now()}.wav`);  // Use .wav extension
      formData.append('language', 'en');  // Add any other parameters, such as 'language'


      // FIXME: TEMP SHUTDOWN

      // // Send the WAV file to the API endpoint via a POST request
      // const response = await fetch('https://hqxcjph1-5000.inc1.devtunnels.ms/process-audio', {
      //   method: 'POST',
      //   body: formData,  // Attach FormData (WAV file) to the request
      // });

      // // Handle the response from the API
      // if (response.ok) {
      //   const result = await response.json();
      //   console.log('API response:', result);
      // } else {
      //   console.error('Failed to upload audio:', response.status, response.statusText);
      // }
    } catch (error) {
      console.error('Error processing audio:', error);
    }
  } else {
    console.warn('No audio chunks available to send.');
  }
};


  // const saveAndSendAudio = async () => {  
  //   if (audioChunks.current.length > 0) {  
  //     try {  
  //       // Create a new AudioContext to decode the audio data  
  //       const audioContext = new (window.AudioContext)();  
  //       const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });  
  //       const arrayBuffer = await audioBlob.arrayBuffer();  
  //       const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);  
        
  //       console.log("audio buffer", audioBuffer)
  //       // Convert to WAV format  
  //       const wavBlob = bufferToWave(audioBuffer);  
  //       audioChunks.current = []; // Clear the chunks for the next recording segment  
  //       downloadWavBlob(wavBlob, 'audio_output.wav');
        
  //       console.log("wave blob", wavBlob)


  //       const formData = new FormData();  
  //       formData.append('audio', wavBlob, `audio_${Date.now()}.wav`);  
  //       formData.append('language', 'en');  
  

  //       const response = await fetch('https://your-api-endpoint.com/process-audio', {  
  //         method: 'POST',  
  //         body: formData,  
  //       });  
  
  //       if (response.ok) {  
  //         const result = await response.json();  
  //         console.log('API response:', result);  
  //       } else {  
  //         console.error('Failed to upload audio:', response.status, response.statusText);  
  //       }  
  //     } catch (error) {  
  //       console.error('Error uploading audio:', error);  
  //     }  
  //   } else {  
  //     console.warn('No audio chunks available to send.');  
  //   }  
  // };

  const CallLayout = () => {
    switch (layout) {
      case 'grid':
        return <PaginatedGridLayout />;
      case 'speaker-right':
        return <SpeakerLayout participantsBarPosition="left" />;
      default:
        return <SpeakerLayout participantsBarPosition="right" />;
    }
  };

  const fetchAnswer = async() => {
    try {
      const data = {
        filename: "Recording1",
        query
      }
      const response = await axios.post(`https://hqxcjph1-5000.inc1.devtunnels.ms/query`, data);
      if(response){
        setQueryRes((prev) => [...prev, response?.data?.answer]);
      }
    } catch (error) {
      console.warn('Error fetching answer...try again');
    }
  }

  if (callingState !== CallingState.JOINED) return <Loader />;

  

  return (
    <section className="relative h-screen w-full overflow-hidden pt-4 text-white">
      <div className="relative flex size-full items-center justify-center">
        <div className="flex size-full max-w-[1000px] items-center">
          <CallLayout />
        </div>
        <div
          className={cn('h-[calc(100vh-86px)] hidden ml-2', {
            'show-block': showParticipants,
          })}
        >
          <CallParticipantsList onClose={() => setShowParticipants(false)} />
        </div>

        <div
          className={cn('h-[calc(100vh-86px)] hidden ml-2', {
            'show-block': showChatbot,
          })}
        >
          <div id="container" className='bg-[#19232d] p-4 rounded-lg h-[80vh]'>
            <div id="header" className='flex gap2 justify-between items-center'>
              <div id="label">Chatbot</div>
              <div id="label" onClick={() => setShowChatbot(false)} className='cursor-pointer font-semibold text-red-500'>
                close
              </div>
            </div>
            <div id="answers" className='h-[68vh] mt-2 overflow-scroll hide-scrollbar'>
              {
                queryRes.map((answer, index) => (
                  <div key={index} className='bg-[#161925] p-4 rounded-t-lg rounded-br-lg'>
                    <div className='text-white'>{answer}</div>
                  </div>
                ))
              }
            </div>
            <div id="questions" className='flex items-center gap-2'>
              <input 
                type="text"
                id="question"
                className='p-2 rounded-lg w-[70%] text-black'
                placeholder='Ask a question'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <div id="askbutton" className=''>
                <button onClick={() => {fetchAnswer();}} className='cursor-pointer rounded-2xl bg-[#161925] p-2 font-semibold hover:bg-[#4c535b]'>
                  ask chat
                </button>
              </div>
            </div>
          </div>
          {/* <CallParticipantsList onClose={() => setShowChatbot(false)} /> */}
        </div>

        <div
          className={cn('h-[calc(100vh-86px)] hidden ml-2', {
            'show-block': showChat,
          })}
        >
          <div id="container" className='bg-[#19232d] p-4 rounded-lg h-[80vh]'>
            <div id="header" className='flex gap2 justify-between items-center'>
              <div id="label">Chat</div>
              <div id="label" onClick={() => setShowChat(false)} className='cursor-pointer font-semibold text-red-500'>
                close
              </div>
            </div>
          </div>
          {/* <CallParticipantsList onClose={() => setShowChatbot(false)} /> */}
        </div>
      </div>
      {/* video layout and call controls */}
      <div className="fixed bottom-0 flex w-full items-center justify-center gap-5">
        <CallControls onLeave={() => router.push(`/`)} />

        <DropdownMenu>
          <div className="flex items-center">
            <DropdownMenuTrigger className="cursor-pointer rounded-2xl bg-[#19232d] px-4 py-2 hover:bg-[#4c535b]  ">
              <LayoutList size={20} className="text-white" />
            </DropdownMenuTrigger>
          </div>
          <DropdownMenuContent className="border-dark-1 bg-dark-1 text-white">
            {['Grid', 'Speaker-Left', 'Speaker-Right'].map((item, index) => (
              <div key={index}>
                <DropdownMenuItem
                  onClick={() =>
                    setLayout(item.toLowerCase() as CallLayoutType)
                  }
                >
                  {item}
                </DropdownMenuItem>
                <DropdownMenuSeparator className="border-dark-1" />
              </div>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <CallStatsButton />

        {/* Chatbot */}
        <button onClick={() => setShowChatbot((prev) => !prev)} className='cursor-pointer rounded-2xl bg-[#19232d] px-4 py-2 hover:bg-[#4c535b]'>
          Open Chatbot
        </button>

        <button onClick={() => setShowChat(true)} className='cursor-pointer rounded-2xl bg-[#19232d] px-4 py-2 hover:bg-[#4c535b]'>
          <MessageCircle size={20} className="text-white" />
        </button>

        <button onClick={() => setShowParticipants((prev) => !prev)}>
          <div className=" cursor-pointer rounded-2xl bg-[#19232d] px-4 py-2 hover:bg-[#4c535b]  ">
            <Users size={20} className="text-white" />
          </div>
        </button>
        {!isPersonalRoom && <EndCallButton />}
      </div>
    </section>
  );
};

export default MeetingRoom;

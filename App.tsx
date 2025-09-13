import React, { useState, useCallback, useEffect } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { ImageUpload } from './components/ImageUpload';
import { ScriptInput } from './components/ScriptInput';
import { LoadingIndicator } from './components/LoadingIndicator';
import { VideoPlayer } from './components/VideoPlayer';
import { VoiceSettings } from './components/VoiceSettings';
import { generateLipSyncedVideo } from './services/geminiService';
import { loadElevenLabsVoices, generateElevenLabsSpeech } from './services/elevenLabsService';
import { LOADING_MESSAGES } from './constants';
import { AppState, ElevenLabsVoice } from './types';

const App: React.FC = () => {
  // Core state
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(null);
  const [script, setScript] = useState<string>('');
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [error, setError] = useState<string | null>(null);

  // Result state
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [elevenLabsAudioUrl, setElevenLabsAudioUrl] = useState<string | null>(null);
  const [loadingMessage, setLoadingMessage] = useState<string>('');

  // ElevenLabs state
  const [elevenLabsApiKey, setElevenLabsApiKey] = useState<string>('');
  const [voices, setVoices] = useState<ElevenLabsVoice[]>([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(null);
  const [isLoadingVoices, setIsLoadingVoices] = useState<boolean>(false);
  const [voiceLoadError, setVoiceLoadError] = useState<string | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (appState === AppState.LOADING) {
      interval = setInterval(() => {
        setLoadingMessage(prev => {
          const currentIndex = LOADING_MESSAGES.indexOf(prev);
          const nextIndex = (currentIndex + 1) % LOADING_MESSAGES.length;
          return LOADING_MESSAGES[nextIndex];
        });
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [appState]);

  const handleFilesChange = (files: File[]) => {
    setImageFiles(files);
    setSelectedImageIndex(files.length > 0 ? 0 : null);
    handleResetResults();
  };

  const handleImageSelect = (index: number) => {
    setSelectedImageIndex(index);
  };
  
  const handleScriptChange = (newScript: string) => {
    setScript(newScript);
    handleResetResults();
  }

  const handleLoadVoices = useCallback(async () => {
    if (!elevenLabsApiKey) {
      setVoiceLoadError('Please enter your ElevenLabs API key.');
      return;
    }
    setIsLoadingVoices(true);
    setVoiceLoadError(null);
    setVoices([]);
    setSelectedVoiceId(null);
    try {
      const loadedVoices = await loadElevenLabsVoices(elevenLabsApiKey);
      setVoices(loadedVoices);
    } catch (err) {
      setVoiceLoadError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoadingVoices(false);
    }
  }, [elevenLabsApiKey]);
  
  const handleGenerateClick = useCallback(async () => {
    if (imageFiles.length === 0 || selectedImageIndex === null || !script) {
      setError('Please provide images, select one, and write a script.');
      return;
    }
    const imageToGenerate = imageFiles[selectedImageIndex];
    if (!imageToGenerate) {
        setError('Selected image is not valid. Please try again.');
        return;
    }

    setError(null);
    handleResetResults();
    setAppState(AppState.LOADING);
    setLoadingMessage(LOADING_MESSAGES[0]);

    let generatedVideoUrl: string | null = null;
    let generatedAudioUrl: string | null = null;

    try {
        const videoPromise = generateLipSyncedVideo(imageToGenerate, script, (msg) => setLoadingMessage(msg));
        const audioPromise = selectedVoiceId && elevenLabsApiKey 
            ? generateElevenLabsSpeech(elevenLabsApiKey, selectedVoiceId, script)
            : Promise.resolve(null);
            
        [generatedVideoUrl, generatedAudioUrl] = await Promise.all([videoPromise, audioPromise]);
        
        if (generatedVideoUrl) {
            setVideoUrl(generatedVideoUrl);
        } else {
             throw new Error("Video generation failed to produce a valid URL.");
        }

        if (generatedAudioUrl) {
            setElevenLabsAudioUrl(generatedAudioUrl);
        }
        
        setAppState(AppState.SUCCESS);

    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during generation.');
      setAppState(AppState.ERROR);
       // Clean up any URLs that might have been created before the error
      if(generatedVideoUrl) URL.revokeObjectURL(generatedVideoUrl);
      if(generatedAudioUrl) URL.revokeObjectURL(generatedAudioUrl);
    }
  }, [imageFiles, selectedImageIndex, script, selectedVoiceId, elevenLabsApiKey]);
  
  const handleResetResults = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    if (elevenLabsAudioUrl) URL.revokeObjectURL(elevenLabsAudioUrl);
    setVideoUrl(null);
    setElevenLabsAudioUrl(null);
    setError(null);
    if(appState !== AppState.LOADING) {
        setAppState(AppState.IDLE);
    }
  }

  const handleFullReset = () => {
    setImageFiles([]);
    setSelectedImageIndex(null);
    setScript('');
    handleResetResults();
  };

  const isGenerateDisabled = imageFiles.length === 0 || selectedImageIndex === null || !script || appState === AppState.LOADING;

  const renderContent = () => {
    switch (appState) {
      case AppState.LOADING:
        return <LoadingIndicator title="Generating Your Masterpiece..." message={loadingMessage} />;
      case AppState.SUCCESS:
        return videoUrl ? <VideoPlayer 
                            videoSrc={videoUrl} 
                            audioSrc={elevenLabsAudioUrl}
                            onReset={handleFullReset} 
                          /> : null;
      default:
        return (
          <>
            <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 items-start mb-8">
              <ImageUpload 
                onFilesChange={handleFilesChange}
                files={imageFiles}
                onImageSelect={handleImageSelect}
                selectedIndex={selectedImageIndex}
              />
              <div className="flex flex-col gap-8">
                <ScriptInput script={script} onScriptChange={handleScriptChange} />
                <VoiceSettings 
                    elevenLabsApiKey={elevenLabsApiKey}
                    onApiKeyChange={setElevenLabsApiKey}
                    voices={voices}
                    selectedVoiceId={selectedVoiceId}
                    onVoiceSelect={setSelectedVoiceId}
                    onLoadVoices={handleLoadVoices}
                    isLoadingVoices={isLoadingVoices}
                    voiceLoadError={voiceLoadError}
                />
              </div>
            </div>
            <div className="w-full flex flex-col items-center">
                <button
                    onClick={handleGenerateClick}
                    disabled={isGenerateDisabled}
                    className="bg-indigo-600 text-white font-bold py-3 px-8 rounded-full hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                    Generate Video
                </button>
            </div>
          </>
        );
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center p-4 sm:p-6 lg:p-8 font-sans bg-gray-900">
      <Header />
      <main className="flex-grow w-full max-w-6xl flex flex-col items-center justify-center py-10">
        {error && (
          <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded-lg relative mb-6 w-full max-w-3xl text-center" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        {renderContent()}
      </main>
      <Footer />
    </div>
  );
};

export default App;
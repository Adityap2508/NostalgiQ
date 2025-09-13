export enum AppState {
    IDLE = 'IDLE',
    LOADING = 'LOADING',
    SUCCESS = 'SUCCESS',
    ERROR = 'ERROR'
}

export interface ElevenLabsVoice {
    voice_id: string;
    name: string;
    labels: {
        accent?: string;
        age?: string;
        gender?: string;
    }
}
from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import cv2
import tempfile
import traceback
import subprocess
import speech_recognition as sr
from werkzeug.utils import secure_filename

# Import from core — use module-level access for mutable globals
# so we always read the live value (e.g. translation_engine is set
# to None initially and reassigned by load_models() at startup).
import src.core as core
from src.core import (
    ensure_meitei_mayek, LANGUAGE_NAMES, STT_LANG_CODES,
    DEEP_TRANSLATOR_CODES, ISL_VIDEO_DIR, fast_predict, extract_landmarks_for_model,
    MP_CONFIDENCE_GATE, MODEL_CONFIDENCE_MIN,
    _translate_to_english, _map_words_to_videos, logger
)
from src.models.nlp_grammar import correct_sentence

api_bp = Blueprint('api', __name__)

@api_bp.route('/initialize', methods=['POST'])
def initialize():
    return jsonify({'success': True, 'message': 'System initialized'})

@api_bp.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'components': {'backend': 'online'}})

# --- Sentence Builder Routes ---
current_sentence = []

@api_bp.route('/api/sentence', methods=['GET', 'POST'])
def handle_sentence():
    global current_sentence
    if request.method == 'POST':
        data = request.get_json()
        action = data.get('action')
        if action == 'add':
            sign = data.get('sign')
            if sign: current_sentence.append(sign)
        elif action == 'undo':
            if current_sentence: current_sentence.pop()
        elif action == 'clear':
            current_sentence = []
    
    sentence_text = ' '.join(current_sentence)
    return jsonify({
        'success': True,
        'words': current_sentence,
        'sentence': sentence_text
    })

# --- Translation Route ---
@api_bp.route('/api/translate', methods=['POST'])
def api_translate():
    try:
        data = request.get_json(force=True, silent=False)
        if data is None:
            return jsonify({'status': 'error', 'success': False, 'error': 'Invalid JSON payload'}), 400

        text = data.get('text', '').strip()
        target_lang = data.get('target_lang') or data.get('target_language') or 'assamese'
        direction = data.get('direction', 'en_to_regional')

        if not text:
            return jsonify({'status': 'error', 'success': False, 'error': 'No text provided'}), 400

        if core.translation_engine is None:
            logger.error('[/api/translate] core.translation_engine is None — TranslationModel failed to load at startup.')
            return jsonify({
                'status': 'error',
                'success': False,
                'error': 'Translation API failed: engine not initialised. Check server logs.'
            }), 503

        if direction == 'en_to_regional':
            try:
                translated = core.translation_engine.translate(text, target_lang)
            except Exception as e:
                print(f'[/api/translate] en_to_regional FAILED — {type(e).__name__}: {e}')
                print(traceback.format_exc())
                logger.error(f'[/api/translate] Translation failed for lang="{target_lang}" input="{text}": {type(e).__name__}: {e}')
                return jsonify({'status': 'error', 'success': False, 'message': str(e)}), 500

            if target_lang == 'manipuri':
                translated = ensure_meitei_mayek(translated)
            source = 'English'
            target = LANGUAGE_NAMES.get(target_lang, target_lang)
        else:
            try:
                translated = core.translation_engine.translate_regional_to_english(text, target_lang)
            except Exception as e:
                print(f'[/api/translate] regional_to_en FAILED — {type(e).__name__}: {e}')
                print(traceback.format_exc())
                logger.error(f'[/api/translate] Reverse translation failed for lang="{target_lang}" input="{text}": {type(e).__name__}: {e}')
                return jsonify({'status': 'error', 'success': False, 'message': str(e)}), 500

            source = LANGUAGE_NAMES.get(target_lang, target_lang)
            target = 'English'

        import json as _json
        return current_app.response_class(
            response=_json.dumps({
                'success': True,
                'data': {
                    'original': text, 'translated': translated,
                    'source': source, 'target': target, 'direction': direction
                }
            }, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print(f'[/api/translate] Unhandled exception — {type(e).__name__}: {e}')
        print(traceback.format_exc())
        logger.error(f'[/api/translate] Unhandled: {type(e).__name__}: {e}')
        return jsonify({'status': 'error', 'success': False, 'message': str(e)}), 500

# --- Grammar Route ---
@api_bp.route('/api/correct-and-translate', methods=['POST'])
def api_correct_and_translate():
    try:
        data = request.get_json()
        words = data.get('words', [])
        target_lang = data.get('target_lang', 'assamese')
        if not words:
            return jsonify({'status': 'error', 'success': False, 'error': 'No words provided'}), 400

        corrected = correct_sentence(words)

        if core.translation_engine is None:
            logger.error('[/api/correct-and-translate] core.translation_engine is None.')
            return jsonify({'status': 'error', 'success': False, 'error': 'Translation API failed: engine not initialised'}), 503

        try:
            translated = core.translation_engine.translate(corrected, target_lang)
        except Exception as e:
            print(f'[/api/correct-and-translate] Translation FAILED — {type(e).__name__}: {e}')
            print(traceback.format_exc())
            logger.error(f'[/api/correct-and-translate] Failed for lang="{target_lang}" input="{corrected}": {type(e).__name__}: {e}')
            return jsonify({'status': 'error', 'success': False, 'message': str(e)}), 500

        if target_lang == 'manipuri':
            translated = ensure_meitei_mayek(translated)

        import json as _json
        return current_app.response_class(
            response=_json.dumps({'success': True, 'data': {'corrected': corrected, 'translated': translated}}, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        print(f'[/api/correct-and-translate] Unhandled — {type(e).__name__}: {e}')
        print(traceback.format_exc())
        logger.error(f'[/api/correct-and-translate] Unhandled: {type(e).__name__}: {e}')
        return jsonify({'status': 'error', 'success': False, 'message': str(e)}), 500

# --- Video Inference Pipeline ---
@api_bp.route('/api/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file submitted'})

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    video_file.save(temp_input.name)
    temp_input.close()
    
    temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_mp4.close()

    try:
        subprocess.run(['ffmpeg', '-y', '-i', temp_input.name, '-c:v', 'copy', temp_mp4.name], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        cap = cv2.VideoCapture(temp_mp4.name)
        frames_processed = 0
        predictions = []
        recent_landmarks = []
        
        frame_skip = 5
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = core.hand_detector.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = extract_landmarks_for_model(frame, results)
                if landmarks is not None:
                    pred_class, conf, _ = fast_predict(landmarks)
                    if pred_class and conf >= MP_CONFIDENCE_GATE:
                        predictions.append((pred_class, conf))
            
            frames_processed += 1

        cap.release()
        
        if not predictions:
            return jsonify({'status': 'error', 'message': 'Could not confidently identify any sign.'})
            
        sign_counts = {}
        for pred, conf in predictions:
            if conf >= MODEL_CONFIDENCE_MIN:
                sign_counts[pred] = sign_counts.get(pred, 0) + 1
                
        if not sign_counts:
            return jsonify({'status': 'error', 'message': 'Sign detected but confidence too low.'})
            
        best_sign = max(sign_counts.items(), key=lambda x: x[1])[0]
        corrected_sentence = correct_sentence([best_sign])
        
        return jsonify({
            'success': True,
            'signs': [best_sign],
            'corrected': corrected_sentence,
            'frames_analyzed': frames_processed
        })
        
    except Exception as e:
        logger.error(f"Video process error: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        for f in [temp_input.name, temp_mp4.name]:
            try:
                if os.path.exists(f): os.unlink(f)
            except: pass

# --- Text & Voice Processing ---
@api_bp.route('/api/process-text', methods=['POST'])
def api_process_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'english')
        if not text: return jsonify({'success': False, 'error': 'No text provided'})

        english_text = _translate_to_english(text, source_lang)
        corrected = correct_sentence(english_text.split())
        sequence, corrected_words = _map_words_to_videos(corrected)

        return jsonify({
            'success': True, 'input_text': text, 'source_lang': source_lang,
            'english_text': english_text, 'corrected': corrected,
            'sequence': sequence, 'video_count': sum(1 for s in sequence if s['has_video']),
            'total_words': len(corrected_words)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/api/speech-to-text', methods=['POST'])
def api_speech_to_text():
    if 'audio' not in request.files: return jsonify({'success': False, 'error': 'No audio file'})
    
    audio_file = request.files['audio']
    source_lang = request.form.get('source_lang', 'english')
    lang_code = STT_LANG_CODES.get(source_lang, 'en-US')

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wav_path = tmp.name
    audio_file.save(wav_path)
    tmp.close()

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        regional_text = recognizer.recognize_google(audio_data, language=lang_code)
        
        if source_lang == 'manipuri':
            regional_text = ensure_meitei_mayek(regional_text)
            
        import json
        return current_app.response_class(
            response=json.dumps({'success': True, 'text': regional_text, 'source_lang': source_lang}, ensure_ascii=False),
            status=200, mimetype='application/json'
        )
    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Could not hear clearly'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if os.path.exists(wav_path): os.unlink(wav_path)

@api_bp.route('/api/animate-sentence', methods=['POST'])
def animate_sentence():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence: return jsonify({'success': False, 'error': 'No sentence provided'})
    sequence, _ = _map_words_to_videos(sentence)
    return jsonify({'success': True, 'sequence': sequence})

@api_bp.route('/api/video/<filename>')
def serve_video(filename):
    """Serve ISL animation .mp4 files with proper MIME type and range support.
    
    Performs case-insensitive filename lookup because the video files
    use Title Case (e.g. Hello.mp4) but frontend requests may arrive
    in any case (e.g. hello.mp4).
    """
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({'error': 'Invalid filename'}), 400

    # ── Case-insensitive lookup ──────────────────────────────────────────
    # Build a lowercase → actual filename map for the video directory
    try:
        actual_files = os.listdir(ISL_VIDEO_DIR)
    except FileNotFoundError:
        logger.error(f"ISL_VIDEO_DIR not found: {ISL_VIDEO_DIR}")
        return jsonify({'error': 'Video directory not found'}), 500

    lower_map = {f.lower(): f for f in actual_files}
    resolved = lower_map.get(safe_name.lower())

    if not resolved:
        logger.warning(f"[serve_video] No match for '{safe_name}' in {ISL_VIDEO_DIR}")
        return jsonify({'error': f'Video not found: {safe_name}'}), 404

    return send_from_directory(
        ISL_VIDEO_DIR,
        resolved,
        mimetype='video/mp4',
        conditional=True   # enables Accept-Ranges: bytes for seeking
    )

@api_bp.route('/api/languages')
def api_languages():
    return jsonify({'success': True, 'languages': list(LANGUAGE_NAMES.keys()), 'names': LANGUAGE_NAMES})

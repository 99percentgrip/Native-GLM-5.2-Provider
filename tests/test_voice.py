"""Tests for glm_acp.voice — local Whisper, notification sounds, desktop notifications."""

from __future__ import annotations

import time

import pytest

from glm_acp import voice


def test_is_voice_available_returns_bool():
    """is_voice_available returns a boolean (True if faster-whisper installed)."""
    result = voice.is_voice_available()
    assert isinstance(result, bool)


def test_play_sound_respects_opt_in_default_off(monkeypatch):
    """play_sound is silent by default (GLM_ACP_SOUND not set or 0)."""
    monkeypatch.delenv("GLM_ACP_SOUND", raising=False)
    voice._SOUND_ENABLED = False
    voice._last_sound = 0.0
    wrote = []
    monkeypatch.setattr(voice.sys.stdout, "write", lambda s: wrote.append(s))
    monkeypatch.setattr(voice.sys.stdout, "flush", lambda: None)
    voice.play_sound("success")
    assert wrote == []


def test_play_sound_plays_when_enabled(monkeypatch):
    """play_sound writes terminal bell when GLM_ACP_SOUND=1 and cooldown passed."""
    monkeypatch.setenv("GLM_ACP_SOUND", "1")
    voice._SOUND_ENABLED = True
    voice._last_sound = 0.0
    wrote = []
    monkeypatch.setattr(voice.sys.stdout, "write", lambda s: wrote.append(s))
    monkeypatch.setattr(voice.sys.stdout, "flush", lambda: None)
    voice.play_sound("success")
    assert any("\a" in s for s in wrote)


def test_play_sound_cooldown_prevents_rapid_repeats(monkeypatch):
    """play_sound enforces a 5-second cooldown between sounds."""
    monkeypatch.setenv("GLM_ACP_SOUND", "1")
    voice._SOUND_ENABLED = True
    voice._last_sound = time.monotonic()
    wrote = []
    monkeypatch.setattr(voice.sys.stdout, "write", lambda s: wrote.append(s))
    monkeypatch.setattr(voice.sys.stdout, "flush", lambda: None)
    voice.play_sound("success")
    assert wrote == []


def test_send_notification_disabled_by_env(monkeypatch):
    """send_notification is a no-op when GLM_ACP_NOTIFY=0."""
    monkeypatch.setenv("GLM_ACP_NOTIFY", "0")
    voice._NOTIFY_ENABLED = False
    calls = []
    monkeypatch.setattr(voice, "_notify_linux", lambda *a, **kw: calls.append(a))
    voice.send_notification("title", "msg")
    assert calls == []


def test_send_notification_skips_short_turns(monkeypatch):
    """send_notification skips when turn_duration < MIN_TURN_SECONDS."""
    monkeypatch.setenv("GLM_ACP_NOTIFY", "1")
    voice._NOTIFY_ENABLED = True
    voice._last_notify = 0.0
    calls = []
    monkeypatch.setattr(voice, "_notify_linux", lambda *a, **kw: calls.append(a))
    voice.send_notification("title", "msg", turn_duration=3.0)
    assert calls == []


def test_send_notification_fires_for_long_turns(monkeypatch):
    """send_notification fires when turn_duration >= MIN_TURN_SECONDS."""
    monkeypatch.setenv("GLM_ACP_NOTIFY", "1")
    voice._NOTIFY_ENABLED = True
    voice._last_notify = 0.0
    calls = []

    def fake_notify(*a, **kw):
        calls.append(a)

    monkeypatch.setattr(voice, "_notify_linux", fake_notify)
    monkeypatch.setattr(voice, "_notify_macos", fake_notify)
    monkeypatch.setattr(voice, "_notify_windows", fake_notify)
    voice.send_notification("title", "msg", turn_duration=15.0)
    assert len(calls) == 1


def test_send_notification_rate_limits(monkeypatch):
    """send_notification enforces a 30-second cooldown."""
    monkeypatch.setenv("GLM_ACP_NOTIFY", "1")
    voice._NOTIFY_ENABLED = True
    voice._last_notify = time.monotonic()
    calls = []
    monkeypatch.setattr(voice, "_notify_linux", lambda *a, **kw: calls.append(a))
    voice.send_notification("title", "msg", turn_duration=15.0)
    assert calls == []


def test_voice_recorder_start_without_tools_returns_false(monkeypatch):
    """VoiceRecorder.start returns False when no recorder command is available."""
    monkeypatch.setattr(voice, "_which", lambda tool: False)
    monkeypatch.setattr(voice, "_recorder_command", lambda path: None)
    recorder = voice.VoiceRecorder()
    assert recorder.start() is False
    assert recorder.recording is False


def test_voice_recorder_stop_when_not_recording_returns_none():
    """VoiceRecorder.stop returns None when not recording."""
    recorder = voice.VoiceRecorder()
    assert recorder.stop() is None


def test_suppress_sound_during_recording_resets_cooldown():
    """suppress_sound_during_recording sets _last_sound to now."""
    voice._last_sound = 0.0
    voice.suppress_sound_during_recording()
    assert voice._last_sound > 0


@pytest.mark.asyncio
async def test_transcribe_audio_returns_empty_when_unavailable(monkeypatch):
    """transcribe_audio returns empty string when faster-whisper is not installed."""
    monkeypatch.setattr(voice, "is_voice_available", lambda: False)
    result = await voice.transcribe_audio("/nonexistent.wav")
    assert result == ""

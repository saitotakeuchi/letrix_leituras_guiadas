/**
 * WORDSYNC - Letrix Leituras Guiadas Player
 * Karaoke-style audio player with word synchronization
 *
 * Features:
 * - Word-level highlighting synced to audio
 * - Smooth transitions between words
 * - Keyboard controls (space, arrows)
 * - Mobile-friendly touch support
 * - Click-to-seek on transcript
 */

class WordSyncPlayer {
  constructor(options = {}) {
    // Configuration
    this.options = {
      audioSelector: '#audioElement',
      playBtnSelector: '#playBtn',
      timelineSelector: '#timeline',
      progressSelector: '#progress',
      currentTimeSelector: '#currentTime',
      durationSelector: '#duration',
      transcriptSelector: '#transcript',
      wordSelector: '.word',
      ...options
    };

    // DOM Elements
    this.audio = document.querySelector(this.options.audioSelector);
    this.playBtn = document.querySelector(this.options.playBtnSelector);
    this.timeline = document.querySelector(this.options.timelineSelector);
    this.progress = document.querySelector(this.options.progressSelector);
    this.currentTimeEl = document.querySelector(this.options.currentTimeSelector);
    this.durationEl = document.querySelector(this.options.durationSelector);
    this.transcript = document.querySelector(this.options.transcriptSelector);

    // State
    this.isPlaying = false;
    this.words = [];
    this.currentWordIndex = -1;
    this.previousWordIndex = -1;  // Track previous word for detecting skipped words
    this.animationFrameId = null;  // For requestAnimationFrame loop

    // Initialize
    this.init();
  }

  init() {
    if (!this.audio) {
      console.warn('WordSyncPlayer: Audio element not found');
      return;
    }

    // Collect words with timestamps
    this.collectWords();

    // Bind event listeners
    this.bindEvents();

    // Update duration when loaded
    if (this.audio.readyState >= 2) {
      this.updateDuration();
    }
  }

  collectWords() {
    const wordElements = document.querySelectorAll(this.options.wordSelector);

    this.words = Array.from(wordElements)
      .filter(el => {
        // Skip words without timestamps (unspoken title with no-sync class)
        if (el.classList.contains('no-sync')) return false;
        if (el.dataset.start === undefined || el.dataset.end === undefined) return false;
        const start = parseFloat(el.dataset.start);
        const end = parseFloat(el.dataset.end);
        // Skip words with zero timestamps (unspoken)
        return !isNaN(start) && !isNaN(end) && (start > 0 || end > 0);
      })
      .map(el => ({
        element: el,
        start: parseFloat(el.dataset.start),
        end: parseFloat(el.dataset.end),
        text: el.textContent.trim()
      }))
      .sort((a, b) => a.start - b.start);
  }

  bindEvents() {
    // Play/Pause button
    if (this.playBtn) {
      this.playBtn.addEventListener('click', () => this.togglePlay());
    }

    // Timeline click for seek
    if (this.timeline) {
      this.timeline.addEventListener('click', (e) => this.seekToPosition(e));
    }

    // Audio events
    this.audio.addEventListener('loadedmetadata', () => this.updateDuration());
    // Note: We still use timeupdate for progress bar updates (less critical)
    // but word highlighting uses requestAnimationFrame for precision
    this.audio.addEventListener('timeupdate', () => this.onTimeUpdateProgress());
    this.audio.addEventListener('ended', () => this.onEnded());
    this.audio.addEventListener('play', () => this.onPlay());
    this.audio.addEventListener('pause', () => this.onPause());
    this.audio.addEventListener('seeked', () => this.onSeeked());

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleKeyboard(e));

    // Touch support for timeline
    if (this.timeline) {
      this.timeline.addEventListener('touchstart', (e) => this.handleTouch(e));
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.audio.pause();
    } else {
      this.audio.play().catch(err => {
        console.log('Playback error:', err);
      });
    }
  }

  onPlay() {
    this.isPlaying = true;
    if (this.playBtn) {
      this.playBtn.classList.remove('play');
      this.playBtn.classList.add('playing');
    }
    // Reset words if restarting from the beginning
    if (this.audio.currentTime < 0.1) {
      this.resetWords();
    }
    // Start the high-precision highlight loop
    this.startHighlightLoop();
  }

  onPause() {
    this.isPlaying = false;
    if (this.playBtn) {
      this.playBtn.classList.add('play');
      this.playBtn.classList.remove('playing');
    }
    // Stop the highlight loop
    this.stopHighlightLoop();
  }

  onEnded() {
    this.isPlaying = false;
    if (this.playBtn) {
      this.playBtn.classList.add('play');
      this.playBtn.classList.remove('playing');
    }
    this.stopHighlightLoop();
    // Mark the last active word as read, but don't reset all words.
    // Words stay yellow (read) until the user presses play again.
    if (this.currentWordIndex >= 0 && this.currentWordIndex < this.words.length) {
      const el = this.words[this.currentWordIndex].element;
      el.classList.remove('active', 'highlight');
      el.classList.add('read');
    }
  }

  onSeeked() {
    // When user seeks, immediately update highlighting
    // This handles the case where audio is paused but user seeks
    const currentTime = this.audio.currentTime;
    this.updateWordHighlight(currentTime);
  }

  /**
   * Start the requestAnimationFrame loop for precise word highlighting.
   * This runs at 60fps (16ms intervals) vs timeupdate's 100-250ms.
   */
  startHighlightLoop() {
    if (this.animationFrameId) return; // Already running

    const update = () => {
      if (!this.isPlaying) return;

      const currentTime = this.audio.currentTime;
      this.updateWordHighlight(currentTime);

      this.animationFrameId = requestAnimationFrame(update);
    };

    this.animationFrameId = requestAnimationFrame(update);
  }

  /**
   * Stop the requestAnimationFrame loop.
   */
  stopHighlightLoop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Handle timeupdate for progress bar only (less timing-critical).
   */
  onTimeUpdateProgress() {
    const currentTime = this.audio.currentTime;
    const duration = this.audio.duration || 1;

    // Update progress bar
    if (this.progress) {
      const percent = (currentTime / duration) * 100;
      this.progress.style.width = `${percent}%`;
    }

    // Update time display
    if (this.currentTimeEl) {
      this.currentTimeEl.textContent = this.formatTime(currentTime);
    }
  }

  updateWordHighlight(currentTime) {
    // Find current word using binary search - O(log n)
    const newWordIndex = this.findWordIndex(currentTime);

    // No change needed
    if (newWordIndex === this.currentWordIndex) return;

    // When we're in a gap between words (newWordIndex === -1),
    // just mark the current word as read and keep everything else.
    // Don't touch previousWordIndex so jumped-over detection still works.
    if (newWordIndex === -1) {
      if (this.currentWordIndex >= 0 && this.currentWordIndex < this.words.length) {
        const el = this.words[this.currentWordIndex].element;
        el.classList.remove('active', 'highlight');
        el.classList.add('read');
      }
      this.previousWordIndex = this.currentWordIndex;
      this.currentWordIndex = -1;
      return;
    }

    // Handle jumped-over words (key fix for visual skipping)
    // Mark all words between last known word and current as "read"
    const lastKnown = Math.max(this.currentWordIndex, this.previousWordIndex);
    if (lastKnown >= 0 && newWordIndex > lastKnown + 1) {
      for (let i = lastKnown + 1; i < newWordIndex; i++) {
        if (i < this.words.length) {
          const el = this.words[i].element;
          el.classList.remove('active', 'highlight');
          el.classList.add('read');
        }
      }
    }

    // Remove highlight from previous current word
    if (this.currentWordIndex >= 0 && this.currentWordIndex < this.words.length) {
      const prevEl = this.words[this.currentWordIndex].element;
      prevEl.classList.remove('active', 'highlight');
      prevEl.classList.add('read');
    }

    // Highlight new current word
    const newEl = this.words[newWordIndex].element;
    newEl.classList.add('active', 'highlight');
    newEl.classList.remove('read');
    this.scrollToWord(newEl);

    // Handle actual seek backward (user clicked/dragged backward)
    // Only when both indices are valid words (not gaps)
    if (this.currentWordIndex >= 0 && newWordIndex < this.currentWordIndex) {
      for (let i = newWordIndex + 1; i <= this.currentWordIndex && i < this.words.length; i++) {
        this.words[i].element.classList.remove('read', 'active', 'highlight');
      }
    }

    // Update state
    this.previousWordIndex = this.currentWordIndex;
    this.currentWordIndex = newWordIndex;
  }

  /**
   * Binary search to find current word index - O(log n)
   */
  findWordIndex(time) {
    let low = 0;
    let high = this.words.length - 1;

    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const word = this.words[mid];

      if (time >= word.start && time < word.end) {
        return mid;
      } else if (time < word.start) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }

    // Return -1 if not in any word's range
    // But check if we're past all words
    if (low > 0 && low <= this.words.length) {
      const prevWord = this.words[low - 1];
      if (time >= prevWord.end) {
        // We're in a gap after this word
        return -1;
      }
    }

    return -1;
  }

  scrollToWord(element) {
    // Only auto-scroll if word is out of viewport
    const rect = element.getBoundingClientRect();
    const viewportHeight = window.innerHeight;

    if (rect.top < 100 || rect.bottom > viewportHeight - 100) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }

  resetWords() {
    this.currentWordIndex = -1;
    this.previousWordIndex = -1;
    this.words.forEach(word => {
      word.element.classList.remove('active', 'highlight', 'read');
    });
  }

  seekToPosition(e) {
    if (!this.timeline || !this.audio.duration) return;

    const rect = this.timeline.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percent = clickX / rect.width;
    const seekTime = percent * this.audio.duration;

    this.audio.currentTime = seekTime;
  }

  seekToWord(index) {
    if (index < 0 || index >= this.words.length) return;

    const word = this.words[index];
    this.audio.currentTime = word.start;

    // Start playing if not already
    if (!this.isPlaying) {
      this.audio.play().catch(() => {});
    }
  }

  handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = this.timeline.getBoundingClientRect();
    const touchX = touch.clientX - rect.left;
    const percent = Math.max(0, Math.min(1, touchX / rect.width));
    const seekTime = percent * this.audio.duration;

    this.audio.currentTime = seekTime;
  }

  handleKeyboard(e) {
    // Ignore if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      return;
    }

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        this.togglePlay();
        break;

      case 'ArrowLeft':
        e.preventDefault();
        this.audio.currentTime = Math.max(0, this.audio.currentTime - 5);
        break;

      case 'ArrowRight':
        e.preventDefault();
        this.audio.currentTime = Math.min(
          this.audio.duration,
          this.audio.currentTime + 5
        );
        break;

      case 'ArrowUp':
        e.preventDefault();
        if (this.currentWordIndex > 0) {
          this.seekToWord(this.currentWordIndex - 1);
        }
        break;

      case 'ArrowDown':
        e.preventDefault();
        if (this.currentWordIndex < this.words.length - 1) {
          this.seekToWord(this.currentWordIndex + 1);
        }
        break;

      case 'Home':
        e.preventDefault();
        this.audio.currentTime = 0;
        break;

      case 'End':
        e.preventDefault();
        this.audio.currentTime = this.audio.duration;
        break;
    }
  }

  updateDuration() {
    if (!this.durationEl || !this.audio.duration) return;
    this.durationEl.textContent = this.formatTime(this.audio.duration);
  }

  formatTime(seconds) {
    if (isNaN(seconds) || !isFinite(seconds)) return '0:00';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  // Public API

  /**
   * Set audio source dynamically
   * @param {string} src - Audio file URL
   */
  setSource(src) {
    this.audio.src = src;
    this.audio.load();
    this.resetWords();
  }

  /**
   * Get current playback state
   * @returns {Object} - { isPlaying, currentTime, duration, currentWord }
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      currentTime: this.audio.currentTime,
      duration: this.audio.duration,
      currentWord: this.currentWordIndex >= 0 ? this.words[this.currentWordIndex] : null
    };
  }

  /**
   * Load word data dynamically (for SPA usage)
   * @param {Array} wordsData - Array of { word, start, end } objects
   */
  loadWords(wordsData) {
    // Clear existing words
    if (this.transcript) {
      this.transcript.innerHTML = '';

      wordsData.forEach(data => {
        const span = document.createElement('span');
        span.className = 'word';
        span.textContent = data.word + ' ';
        span.dataset.start = data.start;
        span.dataset.end = data.end;
        this.transcript.appendChild(span);
      });

      this.collectWords();
    }
  }
}

// Auto-initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.wordSyncPlayer = new WordSyncPlayer();
});

// Also support the legacy class name for backwards compatibility
window.LeituraGuiadaPlayer = WordSyncPlayer;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WordSyncPlayer;
}

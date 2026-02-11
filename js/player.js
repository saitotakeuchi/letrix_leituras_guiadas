/**
 * LETRIX LEITURAS GUIADAS - Audio Player com Karaokê
 * Player de áudio customizado com sincronização de texto (karaokê)
 * Baseado no player original do site homolog.meuletrix.com.br
 */

class LeituraGuiadaPlayer {
  constructor() {
    // Elementos do DOM
    this.audio = document.getElementById('audioElement');
    this.playBtn = document.getElementById('playBtn');
    this.timeline = document.getElementById('timeline');
    this.progress = document.getElementById('progress');
    this.currentTimeEl = document.getElementById('currentTime');
    this.durationEl = document.getElementById('duration');

    // Estado
    this.isPlaying = false;
    this.karaokeSpans = [];

    // Inicializar
    this.init();
  }

  init() {
    // Coletar todos os spans com marcação de tempo (classe .word)
    this.collectKaraokeSpans();

    // Event Listeners
    this.bindEvents();

    // Atualizar duração quando carregado
    if (this.audio) {
      this.audio.addEventListener('loadedmetadata', () => {
        this.updateDuration();
      });

      // Fallback se já carregado
      if (this.audio.readyState >= 2) {
        this.updateDuration();
      }
    }
  }

  collectKaraokeSpans() {
    // Coletar todos os spans com data-start e data-end (classe .word)
    const allSpans = document.querySelectorAll('.word[data-start][data-end], span[data-start][data-end]');
    this.karaokeSpans = Array.from(allSpans)
      .filter(span => {
        // Skip words without timestamps (unspoken title with no-sync class)
        if (span.classList.contains('no-sync')) return false;
        const start = parseFloat(span.dataset.start);
        const end = parseFloat(span.dataset.end);
        // Skip words with zero timestamps (unspoken)
        return !isNaN(start) && !isNaN(end) && (start > 0 || end > 0);
      })
      .map(span => ({
        element: span,
        start: parseFloat(span.dataset.start),
        end: parseFloat(span.dataset.end)
      }));

    // Ordenar por tempo de início
    this.karaokeSpans.sort((a, b) => a.start - b.start);
  }

  bindEvents() {
    // Play/Pause
    if (this.playBtn) {
      this.playBtn.addEventListener('click', () => this.togglePlay());
    }

    // Timeline click para seek
    if (this.timeline) {
      this.timeline.addEventListener('click', (e) => this.seek(e));
    }

    // Audio events
    if (this.audio) {
      this.audio.addEventListener('timeupdate', () => this.updateProgress());
      this.audio.addEventListener('ended', () => this.handleEnded());
      this.audio.addEventListener('play', () => this.handlePlay());
      this.audio.addEventListener('pause', () => this.handlePause());
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleKeyboard(e));
  }

  togglePlay() {
    if (!this.audio) return;

    if (this.isPlaying) {
      this.audio.pause();
    } else {
      this.audio.play().catch(err => {
        console.log('Erro ao reproduzir:', err);
      });
    }
  }

  handlePlay() {
    this.isPlaying = true;
    this.playBtn.classList.remove('play');
    this.playBtn.classList.add('playing');
  }

  handlePause() {
    this.isPlaying = false;
    this.playBtn.classList.add('play');
    this.playBtn.classList.remove('playing');
  }

  handleEnded() {
    this.isPlaying = false;
    this.playBtn.classList.add('play');
    this.playBtn.classList.remove('playing');
    this.resetKaraoke();
  }

  updateProgress() {
    if (!this.audio || !this.progress) return;

    const currentTime = this.audio.currentTime;
    const duration = this.audio.duration || 1;
    const progressPercent = (currentTime / duration) * 100;

    this.progress.style.width = `${progressPercent}%`;

    // Atualizar tempo atual
    if (this.currentTimeEl) {
      this.currentTimeEl.textContent = this.formatTime(currentTime);
    }

    // Atualizar karaokê
    this.updateKaraoke(currentTime);
  }

  updateKaraoke(currentTime) {
    this.karaokeSpans.forEach(item => {
      const { element, start, end } = item;

      // Remover classes anteriores
      element.classList.remove('active', 'highlight', 'read');

      if (currentTime >= start && currentTime < end) {
        // Palavra atual - adiciona destaque amarelo
        element.classList.add('active');
        element.classList.add('highlight');
      } else if (currentTime >= end) {
        // Palavra já lida
        element.classList.add('read');
      }
    });
  }

  resetKaraoke() {
    this.karaokeSpans.forEach(item => {
      item.element.classList.remove('active', 'highlight', 'read');
    });
  }

  seek(e) {
    if (!this.audio || !this.timeline) return;

    const rect = this.timeline.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const seekPercent = clickX / width;
    const seekTime = seekPercent * this.audio.duration;

    this.audio.currentTime = seekTime;
  }

  updateDuration() {
    if (!this.audio || !this.durationEl) return;

    const duration = this.audio.duration;
    if (!isNaN(duration)) {
      this.durationEl.textContent = this.formatTime(duration);
    }
  }

  formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  handleKeyboard(e) {
    // Espaço para play/pause
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
      e.preventDefault();
      this.togglePlay();
    }

    // Setas para seek
    if (e.code === 'ArrowLeft') {
      this.audio.currentTime = Math.max(0, this.audio.currentTime - 5);
    }

    if (e.code === 'ArrowRight') {
      this.audio.currentTime = Math.min(this.audio.duration, this.audio.currentTime + 5);
    }
  }

  // Método para definir o áudio dinamicamente
  setAudioSource(src) {
    if (!this.audio) return;

    this.audio.src = src;
    this.audio.load();
    this.resetKaraoke();
  }

  // Método para atualizar os textos do karaokê dinamicamente
  updateKaraokeText(container, textData) {
    if (!container) return;

    // Limpar container
    container.innerHTML = '';

    // Criar spans com marcações de tempo
    textData.forEach(item => {
      const span = document.createElement('span');
      span.className = 'word';
      span.textContent = item.word + ' ';
      span.dataset.start = item.start;
      span.dataset.end = item.end;
      container.appendChild(span);
    });

    // Recoletar spans
    this.collectKaraokeSpans();
  }
}

// Inicializar quando o DOM estiver pronto
document.addEventListener('DOMContentLoaded', () => {
  window.leituraGuiadaPlayer = new LeituraGuiadaPlayer();
});

// Exportar classe para uso externo
if (typeof module !== 'undefined' && module.exports) {
  module.exports = LeituraGuiadaPlayer;
}

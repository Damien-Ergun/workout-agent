import { Component, inject, signal, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './settings.html',
  styleUrl: './settings.css'
})
export class Settings implements OnInit {
  private http = inject(HttpClient);

  data = signal<any>(null);
  loading = signal<boolean>(true);
  error = signal<string | null>(null);

  hevyKeyInput = '';
  hevySessionCookie = '';
  hevyStatus = signal<string>('');
  aiStatus = signal<string>('');
  prefStatus = signal<string>('');
  pushSubscribed = signal<boolean>(false);

  selectedAiProvider = signal<string>('gemini');
  aiKeyInputs: { [key: string]: string } = {};
  aiModelInputs: { [key: string]: string } = {};

  availableGoals = ['strength', 'hypertrophy', 'endurance', 'fat_loss', 'power'];
  selectedGoals = signal<string[]>([]);
  selectedExperience = signal<string>('intermediate');
  constraintsInput = '';

  ngOnInit() {
    this.loadSettings();
  }

  loadSettings() {
    this.http.get('/api/settings').subscribe({
      next: (res: any) => {
        this.data.set(res);
        const prefs = res.user_prefs || {};
        this.selectedAiProvider.set(prefs.preferred_ai || 'gemini');
        this.selectedGoals.set(prefs.goals || []);
        this.selectedExperience.set(prefs.experience_level || 'intermediate');
        this.constraintsInput = (prefs.constraints || []).join('. ');
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Failed to load settings');
        this.loading.set(false);
      }
    });
  }

  selectProvider(id: string) {
    this.selectedAiProvider.set(id);
  }

  toggleGoal(goal: string) {
    const current = this.selectedGoals();
    if (current.includes(goal)) {
      this.selectedGoals.set(current.filter(g => g !== goal));
    } else {
      this.selectedGoals.set([...current, goal]);
    }
  }

  saveKey(provider: string) {
    const key = provider === 'hevy' ? this.hevyKeyInput.trim() : (this.aiKeyInputs[provider] || '').trim();
    const model = (this.aiModelInputs[provider] || '').trim();

    if (!key) {
      if (provider === 'hevy') {
        this.hevyStatus.set('Please enter an API key.');
      } else {
        this.aiStatus.set('Please enter an API key.');
      }
      return;
    }

    this.http.post('/api/settings/key', { provider, api_key: key, model: model || undefined }).subscribe({
      next: () => {
        if (provider === 'hevy') {
          this.hevyStatus.set('Hevy key saved!');
          this.hevyKeyInput = '';
        } else {
          this.aiStatus.set(`${provider} key saved!`);
          this.aiKeyInputs[provider] = '';
        }
        this.loadSettings();
      },
      error: (err: any) => {
        const msg = err.error?.detail || 'Error saving key.';
        if (provider === 'hevy') this.hevyStatus.set(msg);
        else this.aiStatus.set(msg);
      }
    });
  }

  deleteKey(provider: string) {
    if (!confirm(`Remove your ${provider} API key?`)) return;
    this.http.post('/api/settings/key/delete', { provider }).subscribe({
      next: () => {
        this.loadSettings();
      },
      error: (err: any) => {
        alert(err.error?.detail || 'Could not delete key.');
      }
    });
  }

  verifyHevy() {
    this.hevyStatus.set('Verifying...');
    this.http.post('/api/settings/verify-hevy', { api_key: this.hevyKeyInput.trim() || '__stored__' }).subscribe({
      next: (res: any) => {
        if (res.status === 'ok') {
          this.hevyStatus.set(`✓ Connected! ${res.workout_count} workouts found.`);
        } else {
          this.hevyStatus.set(res.detail || 'Verification failed.');
        }
      },
      error: (err: any) => {
        this.hevyStatus.set(err.error?.detail || 'Verification failed.');
      }
    });
  }

  connectHevySession() {
    const authCookie = this.hevySessionCookie.trim();
    if (!authCookie) {
      this.hevyStatus.set('Paste the auth2.0-token cookie value first.');
      return;
    }
    this.hevyStatus.set('Validating the Hevy web session...');
    this.http.post('/api/settings/hevy-session', { auth_cookie: authCookie }).subscribe({
      next: (res: any) => {
        this.hevySessionCookie = '';
        this.hevyStatus.set(`✓ Connected${res.username ? ' as ' + res.username : ''}.`);
        this.loadSettings();
      },
      error: (err: any) => {
        this.hevyStatus.set(err.error?.detail || 'The Hevy web session could not be connected.');
      }
    });
  }

  syncHistory() {
    if (!confirm('This will rebuild your workout history from Hevy. Continue?')) return;
    this.hevyStatus.set('Syncing history from Hevy...');
    this.http.post('/api/settings/sync-history', {}).subscribe({
      next: (res: any) => {
        this.hevyStatus.set(`✓ Synced! ${res.processed} of ${res.workouts_found} workouts processed.`);
      },
      error: (err: any) => {
        this.hevyStatus.set(err.error?.detail || 'Sync failed.');
      }
    });
  }

  savePreferences() {
    const constraints = this.constraintsInput
      ? this.constraintsInput.split(/[.\n]+/).map(s => s.trim()).filter(Boolean)
      : [];

    const payload = {
      goals: this.selectedGoals(),
      constraints,
      experience_level: this.selectedExperience(),
      preferred_ai: this.selectedAiProvider(),
    };

    this.http.post('/api/settings/preferences', payload).subscribe({
      next: () => {
        this.prefStatus.set('Profile saved!');
        setTimeout(() => this.prefStatus.set(''), 4000);
      },
      error: (err: any) => {
        this.prefStatus.set(err.error?.detail || 'Could not save preferences.');
      }
    });
  }

  async subscribeToPush() {
    try {
      const reg = await navigator.serviceWorker.register('/sw.js');
      const vapidKey = this.data()?.vapid_public_key;
      if (!vapidKey) return;
      const sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: vapidKey
      });
      this.http.post('/api/settings/push-subscribe', sub.toJSON()).subscribe({
        next: () => this.pushSubscribed.set(true)
      });
    } catch {
      // Ignored
    }
  }

  async unsubscribeFromPush() {
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      const sub = await reg?.pushManager.getSubscription();
      if (sub) {
        await sub.unsubscribe();
        this.pushSubscribed.set(false);
      }
    } catch {
      // Ignored
    }
  }
}

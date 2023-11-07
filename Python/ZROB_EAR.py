import argparse
import queue
import sounddevice as sd
import sys
import numpy as np

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
         return text


class ZrobEar():
    def __init__(self,wind=3.75):
        self.deaf=False
        self.parser = argparse.ArgumentParser(add_help=False)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        self.args, self.remaining = self.parser.parse_known_args()
        if self.args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        self.parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.parser])
        self.parser.add_argument(
            'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
            help='input channels to plot (default: the first)')
        self.parser.add_argument(
            '-d', '--device', type=int_or_str,
            help='input device (numeric ID or substring)')
        self.parser.add_argument(
            '-w', '--window', type=float, default=wind*1000, metavar='DURATION',
            help='visible time slot (default: %(default)s ms)')
        self.parser.add_argument(
            '-i', '--interval', type=float, default=30,
            help='minimum time between plot updates (default: %(default)s ms)')
        self.parser.add_argument(
            '-b', '--blocksize', type=int, help='block size (in samples)')
        self.parser.add_argument(
            '-r', '--samplerate', type=float, help='sampling rate of audio device')
        self.parser.add_argument(
            '-n', '--downsample', type=int, default=1, metavar='N',
            help='display every Nth sample (default: %(default)s)')
        self.args = self.parser.parse_args(self.remaining)
        
        if any(c < 1 for c in self.args.channels):
            self.parser.error('argument CHANNEL: must be >= 1')
        
        self.mapping = [c - 1 for c in self.args.channels]  # Channel numbers start with 1
        self.q = queue.Queue()

        if self.args.samplerate is None:
            device_info = sd.query_devices(self.args.device, 'input')
            self.args.samplerate = device_info['default_samplerate'] # type: ignore

        #args.window = pretty_midi.PrettyMIDI(midi_name).get_end_time()*1000

        self.length = int(self.args.window * self.args.samplerate / (1000 * self.args.downsample))


    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())


    def listen(self):
        #self.observation=[]
        
        self.observation = np.zeros((self.length, len(self.mapping)))
        stream = sd.InputStream(
            device=self.args.device, channels=max(self.args.channels),
            samplerate=self.args.samplerate, callback=self.callback)

        with stream:
            while True:
                data = self.q.get()
                shift = len(data)
                self.observation = np.roll(self.observation,-shift, axis=0)
                self.observation[-shift:, :] = data
                if self.deaf:
                    break

    def get_obs(self):
        return self.observation
            

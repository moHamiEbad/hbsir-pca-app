from streamlit.components.v1 import html as st_html
import streamlit as st

def notify_done(sound: bool = True, toast: bool = True, desktop_note: bool = False):
    """Show a toast + optional beep + optional desktop notification."""
    if toast:
        st.toast("PCA finished ✅", icon="✅")
    if sound:
        st_html("""
        <script>
        (function(){
          try{
            const AC = window.AudioContext || window.webkitAudioContext;
            const ctx = new AC();
            function beep(freq, dur){
              const o = ctx.createOscillator();
              const g = ctx.createGain();
              o.type = "sine"; o.frequency.value = freq;
              o.connect(g); g.connect(ctx.destination);
              g.gain.setValueAtTime(0.0001, ctx.currentTime);
              g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.01);
              o.start();
              g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur/1000);
              o.stop(ctx.currentTime + dur/1000 + 0.02);
            }
            const t1 = 0;
            setTimeout(()=>beep(880,180), t1);
            setTimeout(()=>beep(660,160), t1+220);
            setTimeout(()=>beep(990,180), t1+420);
          }catch(e){}
        })();
        </script>
        """, height=0)
    if desktop_note:
        st_html("""
        <script>
        (async function(){
          try{
            if (!("Notification" in window)) return;
            if (Notification.permission === "default") await Notification.requestPermission();
            if (Notification.permission === "granted") {
              new Notification("PCA finished ✅", { body: "Your plots are ready." });
            }
          }catch(e){}
        })();
        </script>
        """, height=0)

from __future__ import annotations
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

def get_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))

def search_album(sp: spotipy.Spotify, query: str) -> dict | None:
    results = sp.search(q=query, type="album", limit=1)
    items = results["albums"]["items"]
    return items[0] if items else None

def get_album_tracks(sp: spotipy.Spotify, album_id: str) -> list[dict]:
    tracks, offset = [], 0
    while True:
        page = sp.album_tracks(album_id, limit=50, offset=offset)
        tracks.extend(page["items"])
        if page["next"] is None:
            break
        offset += 50
    return tracks

def get_track_metadata(sp: spotipy.Spotify, track_id: str) -> dict:
    track = sp.track(track_id)
    artists = ", ".join(a["name"] for a in track["artists"])
    return {
        "track_id":    track["id"],
        "track_name":  track["name"],
        "artist":      artists,
        "album":       track["album"]["name"],
        "popularity":  track["popularity"],
        "preview_url": track.get("preview_url"),
    }

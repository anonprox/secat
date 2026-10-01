"""Spotify Web API provider capability catalog.

This module is the Spotify analogue of :mod:`oca.capability.tmdb`: it contains
only provider API semantics and OAS-backed request mechanics.  It contains no
RestBench task ids, gold routes, entity names, expected answers, or benchmark
surface-pattern rules.

The catalog describes both read/query capabilities and a declarative action
layer.  Action declarations contain provider request mechanics only; the generic
intent compiler turns typed action IR into the same evidence-plan steps used by
all providers.
"""
from __future__ import annotations

import os

from .catalog import Capability, CapabilityCatalog
from .types import (
    ALBUM, ARTIST, DEVICE, GENRE, PLAYBACK, PLAYLIST, TRACK, USER,
    SEMANTIC_FIELDS as _ALL_SEMANTIC_FIELDS,
)

RESOURCE_TYPES = frozenset({TRACK, ALBUM, ARTIST, PLAYLIST, USER, PLAYBACK, DEVICE, GENRE})
def _provider_field_view(field, owners):
    """Return the same semantic field with only this provider's owners."""
    keep = {owner: concrete for owner, concrete in field.provider_field.items() if owner in owners}
    return type(field)(field.name, field.value_type, keep, field.aliases)

SEMANTIC_FIELDS = {
    name: _provider_field_view(field, RESOURCE_TYPES)
    for name, field in _ALL_SEMANTIC_FIELDS.items()
    if field.owners & RESOURCE_TYPES
}

# Provider-local language normalization.  These are ordinary resource synonyms,
# not task phrases.  In particular, Spotify's "artist" is not TMDB's "person".
RESOURCE_ALIASES = {
    "tracks": TRACK, "song": TRACK, "songs": TRACK, "music": TRACK,
    "albums": ALBUM,
    "artists": ARTIST, "singer": ARTIST, "singers": ARTIST,
    "playlists": PLAYLIST,
    "users": USER, "profile": USER,
    "player": PLAYBACK, "playback state": PLAYBACK,
    "devices": DEVICE,
    "genres": GENRE,
}

# Provider-declared stable primitive fields for broad "information/details"
# answers.  The generic compiler consumes this declaration without knowing
# Spotify resource names.  Keep this list to fields returned by the canonical
# detail endpoint rather than optional/deprecated affinity metadata.
DETAIL_FIELDS = {
    TRACK: ("name", "duration_ms", "explicit"),
    ALBUM: ("name", "release_date", "album_type", "total_tracks"),
    ARTIST: ("name",),
    PLAYLIST: ("name",),
    USER: ("name",),
    DEVICE: ("name", "is_playing", "volume_percent"),
}

PLACEHOLDER_RESOURCE = {
    "playlist_id": PLAYLIST,
    "user_id": USER,
    # Spotify uses the generic placeholder {id} for tracks/albums/artists. Its
    # owner is established by the typed capability, so it is intentionally not
    # assigned a single resource here.
}


def _cap(semantic_name: str, source: str, relation: str, target: str,
         endpoint: str, record_path: str, **kw) -> Capability:
    return Capability(semantic_name=semantic_name, source_resource=source,
                      relation=relation, target_resource=target,
                      endpoint=endpoint, record_path=record_path, **kw)


CAPABILITIES: list[Capability] = [
    # Playback/current-state relations.
    _cap("playback.track", PLAYBACK, "track", TRACK,
         "/me/player/currently-playing", "item", embedded_in_source=True,
         aliases=("currently playing track", "current track", "song")),
    _cap("playback.device", PLAYBACK, "device", DEVICE,
         "/me/player", "device", embedded_in_source=True,
         aliases=("active device", "current device")),
    _cap("playback.queue", PLAYBACK, "queue", TRACK,
         "/me/player/queue", "queue[*]", id_field="", label_field="",
         aliases=("queued tracks", "queued songs")),
    _cap("playback.recent_tracks", PLAYBACK, "recent tracks", TRACK,
         "/me/player/recently-played", "items[*].track",
         aliases=("recently played", "recent songs", "play history")),
    _cap("playback.devices", PLAYBACK, "devices", DEVICE,
         "/me/player/devices", "devices[*]",
         aliases=("available devices", "players")),

    # Track relations/details.
    _cap("track.detail", TRACK, "detail", TRACK,
         "/tracks/{id}", "$", parent_binding={"id": "source.id"},
         aliases=("details", "info", "information")),
    _cap("track.album", TRACK, "album", ALBUM,
         "/tracks/{id}", "album", parent_binding={"id": "source.id"},
         embedded_in_source=True, aliases=("record", "release")),
    _cap("track.artists", TRACK, "artists", ARTIST,
         "/tracks/{id}", "artists[*]", parent_binding={"id": "source.id"},
         embedded_in_source=True, aliases=("artist", "singers", "performers")),

    # Album relations/details.
    _cap("album.detail", ALBUM, "detail", ALBUM,
         "/albums/{id}", "$", parent_binding={"id": "source.id"},
         aliases=("details", "info", "information", "album information")),
    _cap("album.tracks", ALBUM, "tracks", TRACK,
         "/albums/{id}/tracks", "items[*]", parent_binding={"id": "source.id"},
         aliases=("songs", "track list", "songs in album")),
    _cap("album.artists", ALBUM, "artists", ARTIST,
         "/albums/{id}", "artists[*]", parent_binding={"id": "source.id"},
         embedded_in_source=True, aliases=("artist", "singers", "performers")),

    # Artist relations/details.
    _cap("artist.detail", ARTIST, "detail", ARTIST,
         "/artists/{id}", "$", parent_binding={"id": "source.id"},
         aliases=("details", "info", "information")),
    _cap("artist.albums", ARTIST, "albums", ALBUM,
         "/artists/{id}/albums", "items[*]", parent_binding={"id": "source.id"},
         query_literals={"include_groups": "album", "limit": 50},
         aliases=("album", "releases", "discography")),
    _cap("artist.top_tracks", ARTIST, "top tracks", TRACK,
         "/artists/{id}/top-tracks", "tracks[*]", parent_binding={"id": "source.id"},
         aliases=("top songs", "popular tracks", "popular songs")),
    _cap("artist.related_artists", ARTIST, "related artists", ARTIST,
         "/artists/{id}/related-artists", "artists[*]", parent_binding={"id": "source.id"},
         aliases=("similar artists", "related", "recommend artists")),
    # Development-mode-safe generic song lookup for an artist. Spotify Search
    # accepts the selected artist's name as q; the fixed type discriminator keeps
    # the returned record universe typed as tracks.
    _cap("artist.tracks", ARTIST, "tracks", TRACK,
         "/search", "tracks.items[*]", query_literals={"type": ["track"]},
         query_bindings={"q": "source.name"},
         aliases=("songs", "a song", "music", "tracks by artist")),
    _cap("artist.genres", ARTIST, "genres", GENRE,
         "/artists/{id}", "genres[*]", parent_binding={"id": "source.id"},
         id_field="", label_field="",
         aliases=("genre", "music genres")),
    _cap("genre.tracks", GENRE, "tracks", TRACK,
         "/search", "tracks.items[*]", query_literals={"type": ["track"]},
         query_bindings={"q": "source.value"},
         aliases=("songs", "same genre tracks", "music")),

    # Playlist relations/details.
    _cap("playlist.detail", PLAYLIST, "detail", PLAYLIST,
         "/playlists/{playlist_id}", "$",
         parent_binding={"playlist_id": "source.id"},
         aliases=("details", "info", "information")),
    _cap("playlist.tracks", PLAYLIST, "tracks", TRACK,
         "/playlists/{playlist_id}/tracks", "items[*].track",
         parent_binding={"playlist_id": "source.id"},
         aliases=("songs", "playlist items", "items")),
    _cap("playlist.owner", PLAYLIST, "owner", USER,
         "/playlists/{playlist_id}", "owner",
         parent_binding={"playlist_id": "source.id"}, embedded_in_source=True,
         id_field="id", label_field="display_name", aliases=("user", "creator")),

    # Current-user library/profile relations.  Required Spotify discriminator
    # parameters are intrinsic provider facts and live in the catalog.
    _cap("user.profile", USER, "profile", USER,
         "/me", "$", aliases=("detail", "details", "account", "me")),
    _cap("user.playlists", USER, "playlists", PLAYLIST,
         "/me/playlists", "items[*]", aliases=("my playlists", "playlist")),
    _cap("user.saved_albums", USER, "saved albums", ALBUM,
         "/me/albums", "items[*].album", aliases=("albums", "my albums", "library albums")),
    _cap("user.saved_tracks", USER, "saved tracks", TRACK,
         "/me/tracks", "items[*].track", aliases=("tracks", "songs", "my music", "library tracks")),
    _cap("user.following_artists", USER, "following artists", ARTIST,
         "/me/following", "artists.items[*]", query_literals={"type": "artist"},
         aliases=("followed artists", "following", "artists i follow")),
    _cap("user.top_artists", USER, "top artists", ARTIST,
         "/me/top/{type}", "items[*]", path_literals={"type": "artists"},
         aliases=("favorite artists", "favourite artists", "favorite artist", "top artist")),
    _cap("user.top_tracks", USER, "top tracks", TRACK,
         "/me/top/{type}", "items[*]", path_literals={"type": "tracks"},
         aliases=("favorite tracks", "favourite tracks", "favorite songs", "top songs")),
]


# Entity lookup route specs. The compiler consumes the query parameter name and
# response record root from this provider metadata rather than assuming TMDB's
# /search/* + query convention.
FIND_ROUTES = {
    TRACK: {"endpoint": "/search", "query_param": "q", "query_literals": {"type": ["track"]},
            "record_path": "tracks.items[*]"},
    ALBUM: {"endpoint": "/search", "query_param": "q", "query_literals": {"type": ["album"]},
            "record_path": "albums.items[*]"},
    ARTIST: {"endpoint": "/search", "query_param": "q", "query_literals": {"type": ["artist"]},
             "record_path": "artists.items[*]"},
    PLAYLIST: {"endpoint": "/search", "query_param": "q", "query_literals": {"type": ["playlist"]},
               "record_path": "playlists.items[*]"},
}

# Population route specs. Fixed path/query literals are API semantics, while
# record_path identifies the resource universe returned by the operation.
POPULATION_ROUTES = {
    (USER, "current"): {"endpoint": "/me", "record_path": "$"},
    (PLAYBACK, "current"): {"endpoint": "/me/player/currently-playing", "record_path": "$"},
    (PLAYBACK, "state"): {"endpoint": "/me/player", "record_path": "$"},
    (TRACK, "currently_playing"): {"endpoint": "/me/player/currently-playing", "record_path": "item"},
    (TRACK, "saved"): {"endpoint": "/me/tracks", "record_path": "items[*].track"},
    (TRACK, "top"): {"endpoint": "/me/top/{type}", "path_literals": {"type": "tracks"},
                     "record_path": "items[*]"},
    (TRACK, "recent"): {"endpoint": "/me/player/recently-played", "record_path": "items[*].track"},
    (TRACK, "queue"): {"endpoint": "/me/player/queue", "record_path": "queue[*]"},
    (ALBUM, "saved"): {"endpoint": "/me/albums", "record_path": "items[*].album"},
    (ALBUM, "new_releases"): {"endpoint": "/browse/new-releases", "record_path": "albums.items[*]"},
    (ARTIST, "following"): {"endpoint": "/me/following", "query_literals": {"type": "artist"},
                            "record_path": "artists.items[*]"},
    (ARTIST, "top"): {"endpoint": "/me/top/{type}", "path_literals": {"type": "artists"},
                      "record_path": "items[*]"},
    (PLAYLIST, "mine"): {"endpoint": "/me/playlists", "record_path": "items[*]"},
    (DEVICE, "available"): {"endpoint": "/me/player/devices", "record_path": "devices[*]"},
}

POPULATION_ALIASES = {
    "current": "current", "me": "current", "current user": "current",
    "current profile": "current", "profile": "current",
    "currently playing": "currently_playing", "current track": "currently_playing",
    "playing now": "currently_playing", "now playing": "currently_playing",
    "saved": "saved", "saved tracks": "saved", "saved songs": "saved",
    "saved albums": "saved", "my music": "saved", "library": "saved",
    "following": "following", "followed": "following", "followed artists": "following",
    "top": "top", "top artists": "top", "top artist": "top", "top tracks": "top", "top track": "top",
    "favorite": "top", "favourite": "top", "favorites": "top", "favourites": "top",
    "favorite artist": "top", "favourite artist": "top", "favorite artists": "top", "favourite artists": "top",
    "favorite track": "top", "favourite track": "top", "favorite song": "top", "favourite song": "top",
    "recent": "recent", "recently played": "recent",
    "queue": "queue", "queued": "queue",
    "mine": "mine", "my playlists": "mine", "playlists": "mine",
    "new releases": "new_releases", "newest releases": "new_releases", "new": "new_releases",
    "available": "available", "available devices": "available",
    "state": "state", "playback state": "state",
}

CATALOG = CapabilityCatalog("spotify", CAPABILITIES)


# Stateful action routes. These are provider API semantics, not benchmark recipes.
# Binding expressions use ``source`` for the action target/context and ``input``
# for the entity/entities being acted on. Literal maps copy a typed IR literal
# into the documented request field. ``fanout`` means a collection binding must
# preserve all selected values rather than silently selecting the first record.
ACTION_ROUTES = {
    "create_playlist": {
        "method": "POST", "endpoint": "/users/{user_id}/playlists",
        "source_resource": USER, "result_resource": PLAYLIST, "record_path": "$",
        # The provider operation is scoped to the authenticated/current user when
        # the request does not name another owner.  Declaring this here lets the
        # provider-neutral compiler synthesize that context without guessing from
        # benchmark wording or from the existence of an arbitrary user population.
        "implicit_source_population": "current",
        # If compact semantic IR attaches a track collection directly to playlist
        # creation, the provider declares the unique post-create population action.
        "input_followup_action": "add_tracks",
        "path_bindings": {"user_id": "source.id"},
        "body_literal_fields": {"name": "name", "public": "public",
                                "collaborative": "collaborative", "description": "description"},
    },
    "add_tracks": {
        "method": "POST", "endpoint": "/playlists/{playlist_id}/tracks",
        "source_resource": PLAYLIST, "input_resource": TRACK,
        "path_bindings": {"playlist_id": "source.id"},
        "query_bindings": {"uris": "input.uri"}, "fanout": ["input.uri"],
    },
    "remove_tracks": {
        "method": "DELETE", "endpoint": "/playlists/{playlist_id}/tracks",
        "source_resource": PLAYLIST, "input_resource": TRACK,
        "path_bindings": {"playlist_id": "source.id"},
        "body_bindings": {"tracks": {"ref": "input.uri", "wrap": "uri_objects"}},
        "fanout": ["input.uri"],
    },
    "update_playlist": {
        "method": "PUT", "endpoint": "/playlists/{playlist_id}",
        "source_resource": PLAYLIST, "canonicalize_source": True,
        "path_bindings": {"playlist_id": "source.id"},
        "body_literal_fields": {"name": "name", "public": "public",
                                "collaborative": "collaborative", "description": "description"},
    },
    "enqueue_track": {
        "method": "POST", "endpoint": "/me/player/queue",
        "input_resource": TRACK, "query_bindings": {"uri": "input.uri"},
    },
    "skip_next": {"method": "POST", "endpoint": "/me/player/next"},
    "pause": {"method": "PUT", "endpoint": "/me/player/pause"},
    "set_volume": {
        "method": "PUT", "endpoint": "/me/player/volume",
        "query_literal_fields": {"volume_percent": "volume_percent"},
        # The endpoint accepts an absolute 0..100 integer only.  Natural action
        # IR can legitimately preserve a qualitative direction when the request
        # does not state a number.  Provider metadata converts those ordinary
        # direction aliases to conservative valid absolute settings; the generic
        # compiler merely applies this declaration.
        "literal_aliases": {
            "volume_percent": {
                # Direction-only requests do not state a relative delta and the
                # Spotify endpoint accepts only an absolute 0..100 value.  Use
                # monotonic-safe extrema so "down" can never accidentally raise
                # a currently-low volume (and vice versa).
                "down": 0, "lower": 0, "decrease": 0, "quieter": 0,
                "up": 100, "raise": 100, "increase": 100, "louder": 100,
            }
        },
    },
    "set_repeat": {
        "method": "PUT", "endpoint": "/me/player/repeat",
        "query_literal_fields": {"state": "state"},
    },
    "play_tracks": {
        "method": "PUT", "endpoint": "/me/player/play",
        "input_resource": TRACK, "canonicalize_input": True,
        "body_bindings": {"uris": "input.uri"},
        "fanout": ["input.uri"],
    },
    "play_context": {
        "method": "PUT", "endpoint": "/me/player/play",
        "input_resources": [ALBUM, PLAYLIST],
        "body_bindings": {"context_uri": "input.uri"},
    },
    "resume_playback": {"method": "PUT", "endpoint": "/me/player/play"},
    "save_tracks": {
        "method": "PUT", "endpoint": "/me/tracks", "input_resource": TRACK,
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
    "remove_saved_tracks": {
        "method": "DELETE", "endpoint": "/me/tracks", "input_resource": TRACK,
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
    "save_albums": {
        "method": "PUT", "endpoint": "/me/albums", "input_resource": ALBUM,
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
    "remove_saved_albums": {
        "method": "DELETE", "endpoint": "/me/albums", "input_resource": ALBUM,
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
    "follow_artists": {
        "method": "PUT", "endpoint": "/me/following", "input_resource": ARTIST,
        "query_literals": {"type": "artist"},
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
    "unfollow_artists": {
        "method": "DELETE", "endpoint": "/me/following", "input_resource": ARTIST,
        "query_literals": {"type": "artist"},
        "query_bindings": {"ids": "input.id"}, "fanout": ["input.id"],
    },
}


# Development Mode 2026 removes these capabilities/fields from the planning
# ontology itself. The transport still supports legacy reproduction separately.
_DEV2026_REMOVED_CAPABILITIES = frozenset({
    "artist.top_tracks", "artist.related_artists",
})
_DEV2026_REMOVED_POPULATIONS = frozenset({(ALBUM, "new_releases")})
_DEV2026_REMOVED_FIELD_OWNERS = {
    "popularity": frozenset({TRACK, ALBUM, ARTIST}),
    "followers": frozenset({ARTIST, USER}),
}

def _dev2026_semantic_fields():
    out = {}
    for name, field in SEMANTIC_FIELDS.items():
        removed = _DEV2026_REMOVED_FIELD_OWNERS.get(name, frozenset())
        keep = {owner: concrete for owner, concrete in field.provider_field.items() if owner not in removed}
        if keep:
            out[name] = type(field)(field.name, field.value_type, keep, field.aliases)
    return out

def profile_view(profile: str | None = None):
    """Return a provider declaration filtered to the selected Spotify surface."""
    selected = str(profile or os.environ.get("SPOTIFY_API_PROFILE", "legacy")).strip().lower()
    if selected != "dev2026":
        return {
            "catalog": CATALOG, "semantic_fields": SEMANTIC_FIELDS,
            "population_routes": POPULATION_ROUTES, "action_routes": ACTION_ROUTES,
        }
    caps = [c for c in CAPABILITIES if c.semantic_name not in _DEV2026_REMOVED_CAPABILITIES]
    pops = {k:v for k,v in POPULATION_ROUTES.items() if k not in _DEV2026_REMOVED_POPULATIONS}
    return {
        "catalog": CapabilityCatalog("spotify", caps),
        "semantic_fields": _dev2026_semantic_fields(),
        "population_routes": pops,
        "action_routes": ACTION_ROUTES,
    }

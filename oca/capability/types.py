"""Resource types and semantic field ownership.

The registry prevents a projection from being satisfied by a merely
shape-compatible field on the wrong resource (for example, substituting one date
concept for another). It contains provider-domain semantics, not benchmark
phrasing or expected answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# --------------------------------------------------------------------------
# Resource types
# --------------------------------------------------------------------------

PERSON = "person"
MOVIE = "movie"
TV = "tv"
SEASON = "season"
EPISODE = "episode"
COLLECTION = "collection"
COMPANY = "company"
NETWORK = "network"
KEYWORD = "keyword"
REVIEW = "review"
IMAGE = "image"
GENRE = "genre"
CREDIT = "credit"

# Spotify provider resources. These are provider-domain types, not benchmark
# labels. Keeping them in the shared type universe lets the generic capability
# graph validate cross-provider catalogs while provider ontologies decide which
# subset is exposed for a given runtime.
TRACK = "track"
ALBUM = "album"
ARTIST = "artist"
PLAYLIST = "playlist"
USER = "user"
PLAYBACK = "playback"
DEVICE = "device"

RESOURCE_TYPES = frozenset({
    PERSON, MOVIE, TV, SEASON, EPISODE, COLLECTION, COMPANY,
    NETWORK, KEYWORD, REVIEW, IMAGE, GENRE, CREDIT,
    TRACK, ALBUM, ARTIST, PLAYLIST, USER, PLAYBACK, DEVICE,
})

#: Resources that are "works" (things that get released and reviewed).
WORK_TYPES = frozenset({MOVIE, TV, SEASON, EPISODE})

#: Path placeholder -> resource type it identifies.
PLACEHOLDER_RESOURCE = {
    "person_id": PERSON,
    "movie_id": MOVIE,
    "series_id": TV,
    "tv_id": TV,
    "season_number": SEASON,
    "episode_number": EPISODE,
    "collection_id": COLLECTION,
    "company_id": COMPANY,
    "network_id": NETWORK,
    "review_id": REVIEW,
    "credit_id": CREDIT,
    "playlist_id": PLAYLIST,
    "user_id": USER,
}


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------

STRING = "string"
DATE = "date"
NUMBER = "number"
BOOLEAN = "boolean"
IDENT = "identifier"
ASSET_PATH = "asset_path"
RECORD = "record"


# --------------------------------------------------------------------------
# Semantic fields
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticField:
    """A user-meaningful field, bound to the resource types that may carry it.

    ``provider_field`` maps resource type -> concrete response field. A resource
    type absent from that mapping cannot satisfy the field at all, which is what
    stops ``birth_date`` from being read off a movie.
    """

    name: str
    value_type: str
    provider_field: dict[str, str]
    aliases: frozenset[str] = field(default_factory=frozenset)

    @property
    def owners(self) -> frozenset[str]:
        return frozenset(self.provider_field)

    def field_for(self, resource_type: str) -> str | None:
        return self.provider_field.get(resource_type)


def _f(name: str, value_type: str, provider_field: dict[str, str],
       *aliases: str) -> SemanticField:
    return SemanticField(name, value_type, dict(provider_field),
                         frozenset(a.casefold() for a in aliases))


#: The semantic field registry. Keys are canonical names.
SEMANTIC_FIELDS: dict[str, SemanticField] = {f.name: f for f in [
    # ---- identity / naming -------------------------------------------------
    _f("name", STRING, {
        PERSON: "name", COMPANY: "name", NETWORK: "name", KEYWORD: "name",
        GENRE: "name", COLLECTION: "name",
        TRACK: "name", ALBUM: "name", ARTIST: "name", PLAYLIST: "name",
        USER: "display_name", DEVICE: "name",
    }, "full name", "person name", "who", "display name", "user name", "username"),
    _f("title", STRING, {
        MOVIE: "title", TV: "name", SEASON: "name", EPISODE: "name",
        COLLECTION: "name",
    }, "movie title", "show title", "series name", "film title"),
    _f("id", IDENT, {r: "id" for r in RESOURCE_TYPES}),

    # ---- dates -------------------------------------------------------------
    # The whole point of this file: birth date belongs to a person, full stop.
    _f("birth_date", DATE, {PERSON: "birthday"},
       "birthday", "date of birth", "born", "birth day", "dob"),
    _f("death_date", DATE, {PERSON: "deathday"}, "deathday", "date of death"),
    _f("release_date", DATE, {
        MOVIE: "release_date", TV: "first_air_date",
        SEASON: "air_date", EPISODE: "air_date", ALBUM: "release_date",
    }, "released", "release day", "air date", "first air date", "premiere"),
    _f("last_air_date", DATE, {TV: "last_air_date"}, "final air date", "ended"),

    # ---- descriptive -------------------------------------------------------
    _f("overview", STRING, {
        MOVIE: "overview", TV: "overview", SEASON: "overview",
        EPISODE: "overview", COLLECTION: "overview",
    }, "description", "synopsis", "plot", "summary"),
    _f("biography", STRING, {PERSON: "biography"}, "bio", "about"),
    _f("birth_place", STRING, {PERSON: "place_of_birth"},
       "place of birth", "born in", "hometown"),
    _f("tagline", STRING, {MOVIE: "tagline", TV: "tagline"}),
    _f("status", STRING, {MOVIE: "status", TV: "status"}),
    _f("homepage", STRING, {
        MOVIE: "homepage", TV: "homepage", PERSON: "homepage",
        COMPANY: "homepage", NETWORK: "homepage",
    }, "website", "official site", "official website"),
    _f("headquarters", STRING, {COMPANY: "headquarters", NETWORK: "headquarters"},
       "headquarter", "head office", "based in", "founded", "where founded",
       "location of company"),
    _f("description", STRING, {COMPANY: "description"}, "company description"),

    # ---- numeric -----------------------------------------------------------
    _f("rating", NUMBER, {
        MOVIE: "vote_average", TV: "vote_average", EPISODE: "vote_average",
    }, "vote average", "score", "average vote", "rated"),
    _f("vote_count", NUMBER, {MOVIE: "vote_count", TV: "vote_count"},
       "number of votes", "votes"),
    _f("popularity", NUMBER, {
        MOVIE: "popularity", TV: "popularity", PERSON: "popularity",
        TRACK: "popularity", ALBUM: "popularity", ARTIST: "popularity",
    }, "popular", "popularity score"),
    _f("runtime", NUMBER, {MOVIE: "runtime", EPISODE: "runtime"},
       "duration", "length", "how long"),
    _f("budget", NUMBER, {MOVIE: "budget"}, "cost"),
    _f("revenue", NUMBER, {MOVIE: "revenue"}, "box office", "gross", "earnings"),
    _f("season_count", NUMBER, {TV: "number_of_seasons"},
       "number of seasons", "how many seasons"),
    _f("episode_count", NUMBER, {TV: "number_of_episodes", SEASON: "episode_count"},
       "number of episodes", "how many episodes"),
    _f("season_number", NUMBER, {SEASON: "season_number", EPISODE: "season_number"}),
    _f("episode_number", NUMBER, {EPISODE: "episode_number"}),

    # ---- Spotify/provider-neutral media metadata --------------------------
    _f("duration_ms", NUMBER, {TRACK: "duration_ms"},
       "duration milliseconds", "track duration ms"),
    _f("explicit", BOOLEAN, {TRACK: "explicit"}, "explicit content"),
    _f("album_type", STRING, {ALBUM: "album_type"}, "album type"),
    _f("total_tracks", NUMBER, {ALBUM: "total_tracks"},
       "track count", "number of tracks", "tracks count"),
    _f("followers", NUMBER, {
        ARTIST: "followers.total", PLAYLIST: "followers.total", USER: "followers.total",
    }, "follower count", "number of followers"),
    _f("uri", STRING, {
        TRACK: "uri", ALBUM: "uri", ARTIST: "uri", PLAYLIST: "uri", USER: "uri",
    }, "spotify uri"),
    _f("is_playing", BOOLEAN, {PLAYBACK: "is_playing", DEVICE: "is_active"},
       "playing", "active playback", "is active"),
    _f("volume_percent", NUMBER, {DEVICE: "volume_percent"},
       "volume", "volume percent"),

    # ---- language / origin -------------------------------------------------
    _f("original_language", STRING, {
        MOVIE: "original_language", TV: "original_language",
    }, "language", "spoken language"),
    _f("origin_country", STRING, {TV: "origin_country", COMPANY: "origin_country"},
       "country", "country of origin"),

    # ---- credit-local fields (live on a credit record, not on a person) ----
    _f("character", STRING, {CREDIT: "character"}, "role", "played", "plays"),
    _f("job", STRING, {CREDIT: "job"}, "position"),
    _f("department", STRING, {CREDIT: "department"}),

    # ---- assets ------------------------------------------------------------
    _f("poster", ASSET_PATH, {
        MOVIE: "poster_path", TV: "poster_path", SEASON: "poster_path",
        COLLECTION: "poster_path",
    }, "poster path", "cover image", "cover", "cover art"),
    _f("backdrop", ASSET_PATH, {
        MOVIE: "backdrop_path", TV: "backdrop_path", COLLECTION: "backdrop_path",
    }, "backdrop path", "background image"),
    _f("profile_image", ASSET_PATH, {PERSON: "profile_path"},
       "profile path", "profile picture", "headshot", "photo of"),
    _f("logo", ASSET_PATH, {COMPANY: "logo_path", NETWORK: "logo_path"},
       "logo path", "brand image"),
    _f("still", ASSET_PATH, {EPISODE: "still_path"}, "still path", "episode image"),
    _f("file_path", ASSET_PATH, {IMAGE: "file_path"}, "image path", "image file"),

    # ---- review ------------------------------------------------------------
    _f("review_author", STRING, {REVIEW: "author"}, "author", "reviewer"),
    _f("review_content", STRING, {REVIEW: "content"}, "review text", "content"),
]}


#: Reverse index: alias -> canonical semantic field name.
_ALIAS_INDEX: dict[str, str] = {}
for _sf in SEMANTIC_FIELDS.values():
    _ALIAS_INDEX[_sf.name.casefold()] = _sf.name
    _ALIAS_INDEX[_sf.name.replace("_", " ").casefold()] = _sf.name
    for _a in _sf.aliases:
        _ALIAS_INDEX.setdefault(_a, _sf.name)


def canonical_field(text: str) -> str | None:
    """Map free-form field wording to a canonical semantic field name.

    Matching is exact-then-token-subset. It deliberately refuses to guess: an
    unrecognised field returns ``None`` so the caller can fail the compile
    rather than fuzzy-match onto whatever looks closest.
    """
    if not text:
        return None
    raw = str(text).strip().casefold()
    if raw in _ALIAS_INDEX:
        return _ALIAS_INDEX[raw]
    squashed = raw.replace("_", " ").replace("-", " ")
    squashed = " ".join(squashed.split())
    if squashed in _ALIAS_INDEX:
        return _ALIAS_INDEX[squashed]
    # Token-subset match, longest alias wins. "date of birth of the director"
    # must reach birth_date without release_date ever being a candidate.
    tokens = set(squashed.split())
    best: tuple[int, str] | None = None
    for alias, canon in _ALIAS_INDEX.items():
        atoks = set(alias.split())
        if atoks and atoks <= tokens:
            score = len(atoks)
            if best is None or score > best[0]:
                best = (score, canon)
    return best[1] if best else None


class FieldTypeError(ValueError):
    """Raised when a semantic field cannot be carried by a resource type."""


def resolve_field(semantic: str, resource_type: str) -> tuple[SemanticField, str]:
    """Resolve ``semantic`` against ``resource_type``.

    Raises :class:`FieldTypeError` when the resource type cannot carry the
    field. This is the hard guard: no scoring, no nearest neighbour.
    """
    canon = canonical_field(semantic)
    if canon is None:
        raise FieldTypeError(f"unknown semantic field: {semantic!r}")
    sf = SEMANTIC_FIELDS[canon]
    provider = sf.field_for(resource_type)
    if provider is None:
        raise FieldTypeError(
            f"semantic field {canon!r} is not carried by resource type "
            f"{resource_type!r}; valid owners: {sorted(sf.owners)}"
        )
    return sf, provider


def compatible_operands(pairs: Iterable[tuple[str, str]]) -> bool:
    """True when every ``(semantic_field, resource_type)`` pair agrees on type.

    Used by the verifier before certifying a comparison. Comparing a person's
    birth date with a movie's release date fails here even though both are
    dates, because the semantic fields differ.
    """
    seen: set[tuple[str, str]] = set()
    for semantic, resource_type in pairs:
        canon = canonical_field(semantic)
        if canon is None:
            return False
        sf = SEMANTIC_FIELDS[canon]
        if sf.field_for(resource_type) is None:
            return False
        seen.add((canon, sf.value_type))
    if not seen:
        return False
    return len({vt for _, vt in seen}) == 1 and len({c for c, _ in seen}) == 1

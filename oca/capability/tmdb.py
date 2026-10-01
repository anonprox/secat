"""TMDB provider capability catalog.

Every entry states a fact about the TMDB API that is documented in its own
specification: where credits live, that crew carries a ``job`` discriminator,
that a TV detail response embeds its networks. None of it references a task id,
an expected answer, or a benchmark route hint.

Adding a provider means adding a file like this one. The compiler is unchanged.
"""

from __future__ import annotations

from .catalog import AUTHORITATIVE, Capability, CapabilityCatalog
from .types import (
    COLLECTION, COMPANY, CREDIT, EPISODE, IMAGE, KEYWORD, MOVIE, NETWORK,
    PERSON, REVIEW, SEASON, TV, GENRE,
    SEMANTIC_FIELDS as _ALL_SEMANTIC_FIELDS,
    PLACEHOLDER_RESOURCE as _ALL_PLACEHOLDER_RESOURCE,
)

RESOURCE_TYPES = frozenset({
    PERSON, MOVIE, TV, SEASON, EPISODE, COLLECTION, COMPANY, NETWORK,
    KEYWORD, REVIEW, IMAGE, GENRE, CREDIT,
})
SEMANTIC_FIELDS = _ALL_SEMANTIC_FIELDS
PLACEHOLDER_RESOURCE = {k: v for k, v in _ALL_PLACEHOLDER_RESOURCE.items()
                        if v in RESOURCE_TYPES}
RESOURCE_ALIASES = {
    "television": TV, "tv show": TV, "tv series": TV, "series": TV, "show": TV,
    "film": MOVIE, "films": MOVIE, "movies": MOVIE,
    "people": PERSON, "persons": PERSON, "actor": PERSON, "director": PERSON,
    "collections": COLLECTION, "companies": COMPANY, "episodes": EPISODE,
    "seasons": SEASON, "networks": NETWORK, "credits": CREDIT, "reviews": REVIEW,
}

_SRC = {"source.id"}


def _cap(semantic_name: str, source: str, relation: str, target: str,
         endpoint: str, record_path: str, **kw) -> Capability:
    return Capability(semantic_name=semantic_name, source_resource=source,
                      relation=relation, target_resource=target,
                      endpoint=endpoint, record_path=record_path, **kw)


CAPABILITIES: list[Capability] = [

    # ======================================================================
    # Movie relations
    # ======================================================================
    _cap("movie.director", MOVIE, "director", PERSON,
         "/3/movie/{movie_id}/credits", "crew[*]",
         parent_binding={"movie_id": "source.id"},
         record_filter={"job": "Director"},
         selection="all", id_field="id", label_field="name",
         aliases=("directed by", "film director", "movie director",
                  "who directed"),
         notes="TMDB exposes directors only through the crew collection of the "
               "credits endpoint, discriminated by job=Director. The movie "
               "detail response carries no director field."),

    _cap("movie.writer", MOVIE, "writer", PERSON,
         "/3/movie/{movie_id}/credits", "crew[*]",
         parent_binding={"movie_id": "source.id"},
         record_filter={"department": "Writing"},
         aliases=("screenwriter", "written by", "script")),

    _cap("movie.cast", MOVIE, "cast", PERSON,
         "/3/movie/{movie_id}/credits", "cast[*]",
         parent_binding={"movie_id": "source.id"},
         selection="all", aliases=("actors", "actresses", "starring", "stars",
                                   "who acted", "performers", "cast members")),

    _cap("movie.lead_actor", MOVIE, "lead actor", PERSON,
         "/3/movie/{movie_id}/credits", "cast[*]",
         parent_binding={"movie_id": "source.id"},
         selection="endpoint_order_first",
         aliases=("leading actor", "main actor", "lead", "protagonist",
                  "main character actor", "star", "leading role", "top billed"),
         notes="TMDB returns cast in billing order; the lead is the first "
               "record. Do not re-rank by popularity."),

    _cap("movie.crew", MOVIE, "crew", PERSON,
         "/3/movie/{movie_id}/credits", "crew[*]",
         parent_binding={"movie_id": "source.id"}),

    _cap("movie.keywords", MOVIE, "keywords", KEYWORD,
         "/3/movie/{movie_id}/keywords", "keywords[*]",
         parent_binding={"movie_id": "source.id"},
         aliases=("keyword", "tags", "tag")),

    _cap("movie.similar", MOVIE, "similar", MOVIE,
         "/3/movie/{movie_id}/similar", "results[*]",
         parent_binding={"movie_id": "source.id"},
         label_field="title",
         aliases=("similar movies", "similar films", "like this movie",
                  "related movies")),

    _cap("movie.recommendations", MOVIE, "recommendations", MOVIE,
         "/3/movie/{movie_id}/recommendations", "results[*]",
         parent_binding={"movie_id": "source.id"},
         label_field="title", schema_supplement=True,
         aliases=("recommended", "recommended movies", "recommendation"),
         notes="The bundled TMDB specification documents no response body for "
               "this path. The record shape below is the provider's published "
               "paginated movie list, identical to /movie/{movie_id}/similar."),

    _cap("movie.reviews", MOVIE, "reviews", REVIEW,
         "/3/movie/{movie_id}/reviews", "results[*]",
         parent_binding={"movie_id": "source.id"},
         label_field="author",
         aliases=("review", "user reviews", "comments", "critiques")),

    _cap("movie.images", MOVIE, "images", IMAGE,
         "/3/movie/{movie_id}/images", "posters[*]",
         parent_binding={"movie_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("cover images", "cover image", "posters", "poster images",
                  "cover", "artwork")),

    _cap("movie.backdrops", MOVIE, "backdrops", IMAGE,
         "/3/movie/{movie_id}/images", "backdrops[*]",
         parent_binding={"movie_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("backdrop images", "background images")),

    _cap("movie.release_dates", MOVIE, "release dates", MOVIE,
         "/3/movie/{movie_id}/release_dates", "results[*]",
         parent_binding={"movie_id": "source.id"},
         id_field="iso_3166_1", label_field="iso_3166_1",
         aliases=("regional release dates", "certification",
                  "release date by country")),

    _cap("movie.production_companies", MOVIE, "production company", COMPANY,
         "/3/movie/{movie_id}", "production_companies[*]",
         parent_binding={"movie_id": "source.id"},
         embedded_in_source=True,
         aliases=("production companies", "studio", "studios", "produced by",
                  "company")),

    _cap("movie.genres", MOVIE, "genre", "genre",
         "/3/movie/{movie_id}", "genres[*]",
         parent_binding={"movie_id": "source.id"},
         embedded_in_source=True, aliases=("genres", "category")),

    # NOTE: ``belongs_to_collection`` is present on the movie detail response
    # but the bundled specification gives it no properties, so it cannot be
    # typed. Collection questions resolve through /search/collection instead,
    # which is also what the corpus gold routes use.

    # ======================================================================
    # TV relations
    # ======================================================================
    _cap("tv.creator", TV, "creator", PERSON,
         "/3/tv/{series_id}", "created_by[*]",
         parent_binding={"series_id": "source.id"},
         embedded_in_source=True,
         aliases=("created by", "creators", "show creator")),

    _cap("tv.cast", TV, "cast", PERSON,
         "/3/tv/{series_id}/credits", "cast[*]",
         parent_binding={"series_id": "source.id"},
         aliases=("actors", "actresses", "starring", "stars", "cast members",
                  "who acted")),

    _cap("tv.lead_actor", TV, "lead actor", PERSON,
         "/3/tv/{series_id}/credits", "cast[*]",
         parent_binding={"series_id": "source.id"},
         selection="endpoint_order_first",
         aliases=("leading actor", "main actor", "lead", "protagonist",
                  "star", "leading role", "top billed", "main character actor")),

    _cap("tv.crew", TV, "crew", PERSON,
         "/3/tv/{series_id}/credits", "crew[*]",
         parent_binding={"series_id": "source.id"}),

    _cap("tv.director", TV, "director", PERSON,
         "/3/tv/{series_id}/credits", "crew[*]",
         parent_binding={"series_id": "source.id"},
         record_filter={"job": "Director"},
         aliases=("directed by", "who directed"),
         notes="Series-level credits. A season- or episode-scoped director "
               "question must resolve through season.director instead."),

    _cap("tv.network", TV, "network", NETWORK,
         "/3/tv/{series_id}", "networks[*]",
         parent_binding={"series_id": "source.id"},
         embedded_in_source=True,
         aliases=("networks", "broadcaster", "channel", "aired on",
                  "broadcast network", "tv network"),
         notes="Networks are embedded in the TV detail response. A network is "
               "NOT a production company; do not substitute one for the other."),

    _cap("tv.production_companies", TV, "production company", COMPANY,
         "/3/tv/{series_id}", "production_companies[*]",
         parent_binding={"series_id": "source.id"},
         embedded_in_source=True,
         aliases=("production companies", "studio", "studios", "produced by",
                  "company")),

    _cap("tv.seasons", TV, "seasons", SEASON,
         "/3/tv/{series_id}", "seasons[*]",
         parent_binding={"series_id": "source.id"},
         embedded_in_source=True, aliases=("season list", "all seasons")),

    _cap("tv.genres", TV, "genre", "genre",
         "/3/tv/{series_id}", "genres[*]",
         parent_binding={"series_id": "source.id"},
         embedded_in_source=True, aliases=("genres", "category")),

    _cap("tv.keywords", TV, "keywords", KEYWORD,
         "/3/tv/{series_id}/keywords", "results[*]",
         parent_binding={"series_id": "source.id"},
         aliases=("keyword", "tags", "tag")),

    _cap("tv.similar", TV, "similar", TV,
         "/3/tv/{series_id}/similar", "results[*]",
         parent_binding={"series_id": "source.id"},
         aliases=("similar shows", "similar series", "related shows")),

    _cap("tv.recommendations", TV, "recommendations", TV,
         "/3/tv/{series_id}/recommendations", "results[*]",
         parent_binding={"series_id": "source.id"},
         aliases=("recommended", "recommended shows", "recommendation")),

    _cap("tv.reviews", TV, "reviews", REVIEW,
         "/3/tv/{series_id}/reviews", "results[*]",
         parent_binding={"series_id": "source.id"},
         label_field="author",
         aliases=("review", "user reviews", "comments")),

    _cap("tv.images", TV, "images", IMAGE,
         "/3/tv/{series_id}/images", "posters[*]",
         parent_binding={"series_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("cover images", "cover image", "posters", "cover",
                  "poster images", "artwork")),

    # ---- season / episode scope -------------------------------------------
    _cap("season.credits", SEASON, "credits", PERSON,
         "/3/tv/{series_id}/season/{season_number}/credits", "cast[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "source.season_number"},
         aliases=("season credits", "season cast")),

    _cap("season.director", SEASON, "director", PERSON,
         "/3/tv/{series_id}/season/{season_number}/credits", "crew[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "source.season_number"},
         record_filter={"job": "Director"},
         aliases=("directed by", "who directed"),
         notes="A season-scoped director question must not be answered from "
               "series-level credits or from created_by."),

    _cap("season.lead_actor", SEASON, "lead actor", PERSON,
         "/3/tv/{series_id}/season/{season_number}/credits", "cast[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "source.season_number"},
         selection="endpoint_order_first",
         aliases=("leading actor", "main actor", "lead", "star")),

    _cap("season.episodes", SEASON, "episodes", EPISODE,
         "/3/tv/{series_id}/season/{season_number}", "episodes[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "source.season_number"},
         embedded_in_source=True, aliases=("episode list", "all episodes")),

    _cap("season.images", SEASON, "images", IMAGE,
         "/3/tv/{series_id}/season/{season_number}/images", "posters[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "source.season_number"},
         id_field="file_path", label_field="file_path",
         aliases=("posters", "cover images", "cover image", "cover")),

    _cap("episode.director", EPISODE, "director", PERSON,
         "/3/tv/{series_id}/season/{season_number}/episode/{episode_number}/credits",
         "crew[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "owner.season_number",
                         "episode_number": "source.episode_number"},
         record_filter={"job": "Director"},
         aliases=("directed by", "who directed")),

    _cap("episode.cast", EPISODE, "cast", PERSON,
         "/3/tv/{series_id}/season/{season_number}/episode/{episode_number}/credits",
         "cast[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "owner.season_number",
                         "episode_number": "source.episode_number"},
         aliases=("actors", "actresses", "cast members", "who acted")),

    _cap("episode.lead_actor", EPISODE, "lead actor", PERSON,
         "/3/tv/{series_id}/season/{season_number}/episode/{episode_number}/credits",
         "cast[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "owner.season_number",
                         "episode_number": "source.episode_number"},
         selection="endpoint_order_first",
         aliases=("leading actor", "main actor", "lead", "star", "top billed")),

    _cap("episode.guest_stars", EPISODE, "guest stars", PERSON,
         "/3/tv/{series_id}/season/{season_number}/episode/{episode_number}",
         "guest_stars[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "owner.season_number",
                         "episode_number": "source.episode_number"},
         embedded_in_source=True, aliases=("guest star", "guests")),

    _cap("episode.images", EPISODE, "images", IMAGE,
         "/3/tv/{series_id}/season/{season_number}/episode/{episode_number}/images",
         "stills[*]",
         parent_binding={"series_id": "owner.id",
                         "season_number": "owner.season_number",
                         "episode_number": "source.episode_number"},
         id_field="file_path", label_field="file_path",
         aliases=("stills", "still images", "cover images", "cover image",
                  "screenshots")),

    # ======================================================================
    # Person relations
    # ======================================================================
    _cap("person.movie_credits", PERSON, "movie credits", MOVIE,
         "/3/person/{person_id}/movie_credits", "cast[*]",
         parent_binding={"person_id": "source.id"},
         label_field="title",
         aliases=("movies", "films", "filmography", "acted in", "movie roles",
                  "appeared in", "movies acted in")),

    _cap("person.directed_movies", PERSON, "directed movies", MOVIE,
         "/3/person/{person_id}/movie_credits", "crew[*]",
         parent_binding={"person_id": "source.id"},
         record_filter={"job": "Director"}, label_field="title",
         aliases=("movies directed", "directed films", "films directed",
                  "movies they directed", "directed by them", "directing credits"),
         notes="Person -> movie is the inverse of movie.director and uses the "
               "crew collection of person movie credits."),

    _cap("person.tv_credits", PERSON, "tv credits", TV,
         "/3/person/{person_id}/tv_credits", "cast[*]",
         parent_binding={"person_id": "source.id"},
         aliases=("tv shows", "television credits", "shows", "series",
                  "tv roles", "television shows")),

    _cap("person.directed_tv", PERSON, "directed tv", TV,
         "/3/person/{person_id}/tv_credits", "crew[*]",
         parent_binding={"person_id": "source.id"},
         record_filter={"job": "Director"}, label_field="name",
         aliases=("tv shows directed", "directed tv shows", "television directed",
                  "directed television", "shows directed", "series directed"),
         notes="Person -> directed TV uses the crew collection of person TV credits."),

    _cap("person.images", PERSON, "images", IMAGE,
         "/3/person/{person_id}/images", "profiles[*]",
         parent_binding={"person_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("photos", "profile images", "pictures", "headshots",
                  "profile pictures", "cover image", "cover images")),

    _cap("person.detail", PERSON, "detail", PERSON,
         "/3/person/{person_id}", "$",
         parent_binding={"person_id": "source.id"},
         aliases=("details", "info", "information", "profile", "biography",
                  "birthday", "birth date", "date of birth", "place of birth"),
         notes="Scalar person attributes (birthday, biography, place of birth) "
               "require this request. They are never present on a movie or TV "
               "record, nor on a credit record."),

    # ======================================================================
    # Collection / company / network
    # ======================================================================
    _cap("collection.parts", COLLECTION, "parts", MOVIE,
         "/3/collection/{collection_id}", "parts[*]",
         parent_binding={"collection_id": "source.id"},
         embedded_in_source=True, label_field="title",
         aliases=("movies", "films", "movies in the collection", "entries",
                  "installments", "part")),

    _cap("collection.images", COLLECTION, "images", IMAGE,
         "/3/collection/{collection_id}/images", "posters[*]",
         parent_binding={"collection_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("posters", "cover images", "cover image", "cover")),

    _cap("collection.detail", COLLECTION, "detail", COLLECTION,
         "/3/collection/{collection_id}", "$",
         parent_binding={"collection_id": "source.id"},
         aliases=("details", "info", "information", "overview")),

    _cap("company.images", COMPANY, "images", IMAGE,
         "/3/company/{company_id}/images", "logos[*]",
         parent_binding={"company_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("logos", "logo images", "logo", "brand images")),

    _cap("company.detail", COMPANY, "detail", COMPANY,
         "/3/company/{company_id}", "$",
         parent_binding={"company_id": "source.id"},
         aliases=("details", "info", "information", "headquarters",
                  "founded", "where founded", "homepage")),

    _cap("network.images", NETWORK, "images", IMAGE,
         "/3/network/{network_id}/images", "logos[*]",
         parent_binding={"network_id": "source.id"},
         id_field="file_path", label_field="file_path",
         aliases=("logos", "logo images", "brand images")),

    _cap("network.detail", NETWORK, "detail", NETWORK,
         "/3/network/{network_id}", "$",
         parent_binding={"network_id": "source.id"},
         aliases=("details", "info", "information", "headquarters",
                  "homepage", "origin country")),
]


# --------------------------------------------------------------------------
# Entity lookup ("find") and population routes
# --------------------------------------------------------------------------

#: resource type -> search endpoint. A "find" is never ambiguous in TMDB.
FIND_ROUTES: dict[str, str] = {
    MOVIE: "/3/search/movie",
    TV: "/3/search/tv",
    PERSON: "/3/search/person",
    COLLECTION: "/3/search/collection",
    COMPANY: "/3/search/company",
}

#: (resource type, semantic population) -> endpoint.
#: ``trending`` and ``popular`` are different populations and must not be
#: interchanged; ``latest`` means the newest database record, not the newest
#: release.
POPULATION_ROUTES: dict[tuple[str, str], str] = {
    (MOVIE, "popular"): "/3/movie/popular",
    (MOVIE, "top_rated"): "/3/movie/top_rated",
    (MOVIE, "now_playing"): "/3/movie/now_playing",
    (MOVIE, "upcoming"): "/3/movie/upcoming",
    (MOVIE, "latest"): "/3/movie/latest",
    (MOVIE, "discover"): "/3/discover/movie",
    (TV, "popular"): "/3/tv/popular",
    (TV, "top_rated"): "/3/tv/top_rated",
    (TV, "on_the_air"): "/3/tv/on_the_air",
    (TV, "airing_today"): "/3/tv/airing_today",
    (TV, "latest"): "/3/tv/latest",
    (TV, "discover"): "/3/discover/tv",
    (PERSON, "popular"): "/3/person/popular",
    # TMDB exposes only a mixed-media trending feed.  For a population already
    # constrained to a concrete media resource, use the provider's type-safe
    # ranked feed rather than a route that can legally return another type.
    # This is a provider capability policy, not a benchmark wording rule.
    (MOVIE, "trending"): "/3/movie/popular",
    (TV, "trending"): "/3/tv/popular",
    ("any", "trending"): "/3/trending/all/{time_window}",
}

#: Population wording -> canonical population key.
POPULATION_ALIASES: dict[str, str] = {
    "popular": "popular", "most popular": "popular", "popularity": "popular",
    "top rated": "top_rated", "highest rated": "top_rated",
    "best rated": "top_rated", "top": "top_rated",
    "now playing": "now_playing", "in theaters": "now_playing",
    "currently playing": "now_playing", "currently released": "now_playing",
    "upcoming": "upcoming", "coming soon": "upcoming",
    "on the air": "on_the_air", "currently on the air": "on_the_air",
    "currently airing": "on_the_air", "on air": "on_the_air",
    "airing today": "airing_today", "today": "airing_today",
    "trending": "trending", "most trending": "trending",
    "latest": "latest", "newest added": "latest",
    "discover": "discover",
}


CATALOG = CapabilityCatalog("tmdb", CAPABILITIES)

"""cubbyllm/worlds/wikikg_relations.py -- the wiki world's relations: each relation's noun, its inverse and its form,
and the template facts and hop phrases they give. Split out of wikikg.py (order 0031), verbatim; wikikg.py re-exports
every name.

Wired: WIRED (the wiki world).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


# relation -> (noun, inverse noun or None, form). form "so": "S is the NOUN of O"; "os": "O is the NOUN of S".
# The inverse noun reads the same triple the other way (materialized as a second fact). Symmetric relations
# name themselves as their inverse. Ordered by triple count (validation/logs/exp_g4_wikikg_screen.log).
RELATIONS: dict[str, tuple[str, str | None, str]] = {
    "TIMELINE_EVENT": ("timeline event", None, "os"),
    "SAME_PLACE_AS": ("later name", "earlier name", "os"),        # order 0015: an ancient name in its modern one
    "HAS_PART": ("part", "whole", "os"),
    "MEMBER_OF": ("member", "group", "so"),
    "IS_A": ("type", "instance", "os"),
    "PART_OF": ("part", "whole", "so"),
    "VARIANT_OF": ("variant", "original", "so"),
    "INSTANCE_OF": ("instance", "type", "so"),
    "LOCATED_IN": ("location", None, "os"),
    "PERFORMED_BY": ("performer", "performance", "os"),
    "CREATED_BY": ("creator", "creation", "os"),
    "CAUSED_BY": ("cause", "effect", "os"),
    "LOCATION_OF": ("location", None, "so"),
    "POSITION_HELD": ("position", "holder", "os"),
    "RELATED_TO": ("related entity", "related entity", "os"),
    "INFLUENCED_BY": ("influence", None, "os"),
    "BORN_ON": ("birth date", None, "os"),
    "AUTHORED_BY": ("author", "work", "os"),
    "OCCUPATION": ("occupation", None, "os"),
    "ASSOCIATED_WITH": ("associate", "associate", "os"),
    "SUBCLASS_OF": ("subclass", "superclass", "so"),
    "SPOUSE_OF": ("spouse", "spouse", "so"),
    "BORN_IN": ("birthplace", None, "os"),
    "RELATIVE_OF": ("relative", "relative", "so"),
    "FIELD_OF_WORK": ("field of work", None, "os"),
    "DIED_ON": ("death date", None, "os"),
    "PARENT_OF": ("parent", "child", "so"),
    "COMPOSED_OF": ("component", "whole", "os"),
    "EDUCATED_AT": ("alma mater", None, "os"),
    "AWARDED_TO": ("recipient", "award", "os"),
    "BASED_ON": ("basis", "adaptation", "os"),
    "INSPIRED_BY": ("inspiration", None, "os"),
    "CHILD_OF": ("child", "parent", "so"),
    "CAUSES": ("cause", "effect", "so"),
    "CREATOR_OF": ("creator", "creation", "so"),
    "WORKS_FOR": ("employer", "employee", "os"),
    "COLLABORATED_WITH": ("collaborator", "collaborator", "os"),
    "COMPONENT_OF": ("component", "whole", "so"),
    "AUTHOR_OF": ("author", "work", "so"),
    "PRECEDED_BY": ("predecessor", "successor", "os"),
    "INFLUENCES": ("influence", None, "so"),
    "COMPETED_IN": ("competition", "competitor", "os"),
    "DIED_IN": ("place of death", None, "os"),
    "FOUNDED_BY": ("founder", "founding", "os"),
    "LED_BY": ("leader", None, "os"),
    "FOUNDER_OF": ("founder", "founding", "so"),
    "TYPE_OF": ("type", None, "so"),
    "DERIVED_FROM": ("origin", "derivative", "os"),
    "USED_FOR": ("use", None, "os"),
    "SUCCEEDED_BY": ("successor", "predecessor", "os"),
    "GENERALIZATION_OF": ("generalization", "specialization", "so"),
    "GENRE": ("genre", None, "os"),
    "PRODUCED_BY": ("producer", "production", "os"),
    "CATEGORY_OF": ("category", None, "so"),
    "DIRECTED_BY": ("director", "film", "os"),
    "RESIDED_IN": ("residence", None, "os"),
    "PUBLISHED_BY": ("publisher", "publication", "os"),
    "MANAGED_BY": ("manager", None, "os"),
    "STUDENT_OF": ("student", "teacher", "so"),
    "AFFILIATED_WITH": ("affiliation", "affiliate", "os"),
    "PLAYS_FOR_TEAM": ("team", "player", "os"),
    "RECIPIENT_OF": ("recipient", "award", "so"),
    "WON_MEDAL_AT": ("medal event", None, "os"),
    "ALMA_MATER": ("alma mater", None, "os"),
    "MENTOR_OF": ("mentor", "student", "so"),
    "SIBLING_OF": ("sibling", "sibling", "so"),
    "TRACK_OF": ("track", "album", "so"),
    "ETHNICITY": ("ethnicity", None, "os"),
    "NATIONALITY": ("nationality", None, "os"),
    "AFTER": ("successor", "predecessor", "so"),
    "USED_IN": ("use", None, "os"),
    "COLLEAGUE_OF": ("colleague", "colleague", "so"),
    "DISCOVERED_BY": ("discoverer", "discovery", "os"),
    "BURIED_AT": ("burial place", None, "os"),
    "USES": ("tool", "user", "os"),
    "BORDERS": ("neighbour", "neighbour", "os"),
    "EMPLOYED_BY": ("employer", "employee", "os"),
    "ALBUM_OF": ("album", None, "so"),
    "SUBSET_OF": ("subset", "superset", "so"),
    "CONTEMPORARY_WITH": ("contemporary", "contemporary", "os"),
    "MARRIED_TO": ("spouse", "spouse", "os"),
    "BEFORE": ("predecessor", "successor", "so"),
    "ADAPTED_FROM": ("source", "adaptation", "os"),
    "CAPITAL_OF": ("capital", None, "so"),
    "ALIAS_OF": ("alias", "alias", "so"),
    "USED_BY": ("user", "tool", "os"),
    "AWARDED_RANK": ("rank", None, "os"),
    "HOST_OF": ("host", None, "so"),
    "RELEASED_ON_LABEL": ("label", None, "os"),
    "NAMED_AFTER": ("namesake", None, "os"),
    "KNOWN_FOR": ("claim to fame", None, "os"),
    "INVENTED_BY": ("inventor", "invention", "os"),
    "CITES": ("citation", None, "os"),
    "DURING": ("period", None, "os"),
    "SUBSIDIARY_OF": ("subsidiary", "parent company", "so"),
    "ABOUT": ("subject", None, "os"),
    "CONTRIBUTED_TO": ("contribution", "contributor", "os"),
    "COMMANDER_OF": ("commander", None, "so"),
    "CONTAINS": ("part", "whole", "os"),
    "DIRECTOR_OF": ("director", "film", "so"),
    "SUBJECT_OF": ("subject", None, "so"),
    "SUPPORTED_BY": ("supporter", None, "os"),
    "SPECIALIZATION_OF": ("specialization", "generalization", "so"),
    "SERVED_IN": ("service", None, "os"),
    "OBSERVED_BY": ("observer", None, "os"),
    "ACQUIRED_BY": ("acquirer", "acquisition", "os"),
    "EPISODE_OF": ("episode", "series", "so"),
    "COACH_OF": ("coach", "team", "so"),
    "NOMINATED_FOR": ("nomination", None, "os"),
    "SIMILAR_TO": ("similar entity", "similar entity", "os"),
    "RELIGION": ("religion", None, "os"),
    "ALSO_KNOWN_AS": ("alias", "alias", "os"),
    "APPLIED_TO": ("application", None, "os"),
    "INVOLVED_IN": ("involvement", None, "os"),
    "PARTNER_OF": ("partner", "partner", "so"),
    "INDUSTRY": ("industry", None, "os"),
    "TRANSLATED_BY": ("translator", None, "os"),
    "HEADQUARTERED_IN": ("headquarters", None, "os"),
    "RECORDED_AT": ("recording location", None, "os"),
    "ILLUSTRATED_BY": ("illustrator", None, "os"),
    "FOCUSES_ON": ("focus", None, "os"),
    "TREATS": ("treatment", None, "so"),
    "SUPPORTS": ("supporter", None, "so"),
    "OPPOSED_BY": ("opponent", "opponent", "os"),
    "OPPOSES": ("opponent", "opponent", "os"),
    "DESCRIBED_BY": ("description", None, "os"),
    "INFLUENCED": ("influence", None, "so"),
    "LEADER_OF": ("leader", None, "so"),
    "DIVISION_OF": ("division", None, "so"),
    "OPPOSED_TO": ("opponent", "opponent", "os"),
    "COACHED_BY": ("coach", "team", "os"),
    "MERGED_WITH": ("merger partner", "merger partner", "os"),
    "REQUIRES": ("requirement", None, "os"),
    "EVENT_OF": ("event", None, "so"),
    "STUDIED_BY": ("researcher", None, "os"),
    "OWNER_OF": ("owner", "property", "so"),
    "LANGUAGE_OF": ("language", None, "so"),
    "FLOWS_INTO": ("mouth", None, "os"),
    "DEFINED_BY": ("definition", None, "os"),
    "OWNED_BY": ("owner", "property", "os"),
    "COMPOSED_BY": ("composer", "composition", "os"),
    "DEDICATED_TO": ("dedicatee", None, "os"),
    "RESPONSIBLE_FOR": ("responsibility", None, "os"),
    "PROPERTY_OF": ("property", None, "so"),
    "APPEARS_IN": ("appearance", None, "os"),
    "TOPIC_OF": ("topic", None, "so"),
    "TRIBUTARY_OF": ("tributary", None, "so"),
    "CONVICTED_OF": ("conviction", None, "os"),
    "CAPABLE_OF": ("capability", None, "os"),
    "RESULT_OF": ("result", None, "so"),
    "FUNCTION_OF": ("function", None, "so"),
    "PARTICIPATED_IN": ("event", "participant", "os"),
    "DESIGNED_BY": ("designer", "design", "os"),
    "DEVELOPED_BY": ("developer", "development", "os"),
    "WRITTEN_BY": ("writer", "work", "os"),
    "STARRED_IN": ("film", "cast member", "os"),
    "STARRING": ("cast member", "film", "os"),
    "PLAYED_BY": ("actor", "role", "os"),
    "SPEAKS": ("language", None, "os"),
    "SUCCESSOR_OF": ("successor", "predecessor", "so"),
    "PREDECESSOR_OF": ("predecessor", "successor", "so"),
    "TEACHER_OF": ("teacher", "student", "so"),
    "PARENT_COMPANY_OF": ("parent company", "subsidiary", "so"),
    "CAPITAL": ("capital", None, "os"),
}


_DATE = re.compile(r"[\d\-/.:]+")


def decamel(s: str) -> str:
    """'KeikoMatsuzaka' -> 'Keiko Matsuzaka', 'AlphabetInc_Creation' -> 'Alphabet Inc Creation'; dates untouched."""
    if _DATE.fullmatch(s):
        return s
    s = s.replace("_", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return " ".join(s.split())


def relation_entry(relation: str) -> tuple[str, str | None, str]:
    """The map entry, or the generic verbalization for an unmapped type."""
    if relation in RELATIONS:
        return RELATIONS[relation]
    if relation.endswith("_OF"):
        return (relation[:-3].lower().replace("_", " "), None, "so")
    return (relation.lower().replace("_", " "), None, "os")


def facts_for(subject: str, relation: str, obj: str, inverse: bool = True) -> list[str]:
    """The template fact(s) for one triple: the forward reading, plus the inverse reading when the map names one."""
    s, o = decamel(subject), decamel(obj)
    if not s or not o or s == o:
        return []
    noun, inv, form = relation_entry(relation)
    if form == "so":
        out = [f"{s} is the {noun} of {o}"]
        if inverse and inv:
            out.append(f"{o} is the {inv} of {s}")
    else:
        out = [f"{o} is the {noun} of {s}"]
        if inverse and inv:
            out.append(f"{s} is the {inv} of {o}")
    return out


def hop_phrase(relation: str, direction: str) -> str | None:
    """The question phrase P for walking one hop of a path so that 'next is the P of here' is a stored fact.
    forward = along the triple (subject -> object); backward = against it. None when that reading needs an
    inverse the map does not name."""
    noun, inv, form = relation_entry(relation)
    if direction == "forward":
        return noun if form == "os" else inv
    return noun if form == "so" else inv

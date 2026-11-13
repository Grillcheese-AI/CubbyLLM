"""Order 0072, build step 1: the table holds 67 of 67 verbs, each with its French forms, its kinds and their arguments,
or its act classes with a read-back; no two English verbs share a French infinitive; French imperatives follow their
group's rule, irregular verbs by hand."""
from __future__ import annotations

import pytest

from cubbyllm.speech import verb_kinds as V

ORDER_VERBS = ("add analyze apply build calculate check clear close compare compile configure connect copy create define "
               "delete deploy describe disable download edit enable evaluate execute explain export extract filter find "
               "format generate get identify import install list load merge modify move open parse print process read "
               "remove rename replace reset restart run save search select send set sort start stop summarize test "
               "translate update upload validate verify write").split()


def test_the_table_holds_the_67_verbs_of_the_order():
    assert len(ORDER_VERBS) == 67 and sorted(V.VERBS) == sorted(ORDER_VERBS)


@pytest.mark.parametrize("name", ORDER_VERBS)
def test_every_verb_is_complete(name):
    v = V.VERBS[name]
    assert all(v["fr"][k] for k in ("infinitive", "tu", "nous", "vous"))
    assert v["kinds"] or v["acts"]
    for k in v["kinds"]:
        assert k["kind"] in V.KINDS and k["args"]
    for a in v["acts"]:
        assert a["class"] in V.CLASSES and a["read_back"]


def test_no_two_verbs_share_a_french_infinitive():
    infs = [v["fr"]["infinitive"] for v in V.VERBS.values()]
    assert len(infs) == len(set(infs))


@pytest.mark.parametrize("inf,want", [
    ("trier", ("trie", "trions", "triez")), ("déplacer", ("déplace", "déplaçons", "déplacez")),
    ("charger", ("charge", "chargeons", "chargez")), ("envoyer", ("envoie", "envoyons", "envoyez")),
    ("générer", ("génère", "générons", "générez")), ("régler", ("règle", "réglons", "réglez")),
    ("créer", ("crée", "créons", "créez")), ("définir", ("définis", "définissons", "définissez")),
    ("extraire", ("extrais", "extrayons", "extrayez")), ("mettre à jour", ("mets à jour", "mettons à jour", "mettez à jour")),
    ("télécharger", ("télécharge", "téléchargeons", "téléchargez")), ("sélectionner", ("sélectionne", "sélectionnons", "sélectionnez")),
])
def test_french_imperatives_by_rule(inf, want):
    assert V.imperative(inf) == want


def test_a_verb_with_no_rule_is_refused_not_guessed():
    with pytest.raises(ValueError):
        V.imperative("venir")


def test_acts_ask_first_where_the_order_says():
    ask = {n for n, v in V.VERBS.items() if any(a["class"] == "ask first" for a in v["acts"])}
    assert {"delete", "deploy", "send", "install", "upload", "execute", "run", "remove", "clear", "print"} <= ask
    assert V.VERBS["remove"]["acts"][0]["read_back"] == V.VERBS["delete"]["acts"][0]["read_back"]

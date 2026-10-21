"""Scout every Shakespeare play's Wikipedia plot/synopsis section. Writes scout/plays.jsonl (raw section text
and sentences, not analysed) and prints a table."""
import json
import sys

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, fetch_extract, plot_section, sentences, split_sections, MAX_SENTENCES

# (wikipedia title, short label, First Folio section or note, principal character names used for keeps_names)
PLAYS = [
    ("All's Well That Ends Well", "All's Well", "comedy", ["Helena", "Bertram", "Parolles", "Diana", "Lafeu"]),
    ("As You Like It", "As You Like It", "comedy", ["Rosalind", "Orlando", "Celia", "Touchstone", "Jaques", "Oliver"]),
    ("The Comedy of Errors", "Comedy of Errors", "comedy", ["Antipholus", "Dromio", "Adriana", "Luciana", "Egeon", "Aemilia"]),
    ("Love's Labour's Lost", "Love's Labour's Lost", "comedy", ["Navarre", "Berowne", "Longaville", "Dumaine", "Rosaline", "Boyet"]),
    ("Measure for Measure", "Measure for Measure", "comedy", ["Angelo", "Isabella", "Claudio", "Vincentio", "Mariana", "Lucio"]),
    ("The Merchant of Venice", "Merchant of Venice", "comedy", ["Shylock", "Antonio", "Bassanio", "Portia", "Jessica", "Gratiano"]),
    ("The Merry Wives of Windsor", "Merry Wives", "comedy", ["Falstaff", "Ford", "Page", "Quickly", "Fenton", "Slender"]),
    ("A Midsummer Night's Dream", "Midsummer Night's Dream", "comedy", ["Hermia", "Lysander", "Demetrius", "Helena", "Oberon", "Titania", "Puck", "Bottom"]),
    ("Much Ado About Nothing", "Much Ado", "comedy", ["Beatrice", "Benedick", "Hero", "Claudio", "Don John", "Dogberry"]),
    ("The Taming of the Shrew", "Taming of the Shrew", "comedy", ["Katherina", "Petruchio", "Bianca", "Baptista", "Lucentio", "Hortensio"]),
    ("Twelfth Night", "Twelfth Night", "comedy", ["Viola", "Orsino", "Olivia", "Sebastian", "Malvolio", "Toby"]),
    ("The Two Gentlemen of Verona", "Two Gentlemen", "comedy", ["Valentine", "Proteus", "Julia", "Silvia", "Launce"]),
    ("The Two Noble Kinsmen", "Two Noble Kinsmen", "comedy (not in First Folio; with Fletcher)", ["Palamon", "Arcite", "Emilia", "Theseus"]),
    ("The Winter's Tale", "Winter's Tale", "comedy (romance)", ["Leontes", "Hermione", "Perdita", "Polixenes", "Florizel", "Paulina"]),
    ("Cymbeline", "Cymbeline", "tragedy in Folio (romance)", ["Cymbeline", "Imogen", "Innogen", "Posthumus", "Iachimo", "Cloten"]),
    ("Pericles, Prince of Tyre", "Pericles", "romance (not in First Folio)", ["Pericles", "Marina", "Thaisa", "Antiochus", "Cleon"]),
    ("The Tempest", "The Tempest", "comedy (romance)", ["Prospero", "Miranda", "Ariel", "Caliban", "Ferdinand", "Alonso"]),
    ("Troilus and Cressida", "Troilus and Cressida", "problem play", ["Troilus", "Cressida", "Pandarus", "Achilles", "Hector", "Ulysses"]),
    ("King John (play)", "King John", "history", ["John", "Arthur", "Philip", "Constance", "Hubert", "Pandulph"]),
    ("Richard II (play)", "Richard II", "history", ["Richard", "Bolingbroke", "Gaunt", "York", "Aumerle", "Mowbray"]),
    ("Henry IV, Part 1", "Henry IV Part 1", "history", ["Hal", "Falstaff", "Hotspur", "Henry", "Poins", "Glendower"]),
    ("Henry IV, Part 2", "Henry IV Part 2", "history", ["Hal", "Falstaff", "Henry", "Shallow", "Pistol", "Quickly"]),
    ("Henry V (play)", "Henry V", "history", ["Henry", "Fluellen", "Pistol", "Katherine", "Dauphin", "Exeter"]),
    ("Henry VI, Part 1", "Henry VI Part 1", "history", ["Talbot", "Joan", "Henry", "Gloucester", "Suffolk", "Margaret"]),
    ("Henry VI, Part 2", "Henry VI Part 2", "history", ["Henry", "Gloucester", "Margaret", "Suffolk", "York", "Cade"]),
    ("Henry VI, Part 3", "Henry VI Part 3", "history", ["Henry", "Margaret", "Edward", "Warwick", "Richard", "Clifford"]),
    ("Richard III (play)", "Richard III", "history", ["Richard", "Buckingham", "Clarence", "Anne", "Richmond", "Elizabeth"]),
    ("Henry VIII (play)", "Henry VIII", "history (with Fletcher)", ["Henry", "Wolsey", "Katherine", "Anne", "Buckingham", "Cranmer"]),
    ("Edward III (play)", "Edward III", "history (not in First Folio)", ["Edward", "Countess", "Warwick", "Prince", "John"]),
    ("Titus Andronicus", "Titus Andronicus", "tragedy", ["Titus", "Tamora", "Aaron", "Lavinia", "Saturninus", "Lucius"]),
    ("Romeo and Juliet", "Romeo and Juliet", "tragedy", ["Romeo", "Juliet", "Mercutio", "Tybalt", "Benvolio", "Laurence", "Paris"]),
    ("Julius Caesar (play)", "Julius Caesar", "tragedy", ["Caesar", "Brutus", "Cassius", "Antony", "Portia", "Casca"]),
    ("Hamlet", "Hamlet", "tragedy", ["Hamlet", "Claudius", "Gertrude", "Ophelia", "Polonius", "Laertes", "Horatio"]),
    ("Othello", "Othello", "tragedy", ["Othello", "Iago", "Desdemona", "Cassio", "Emilia", "Roderigo"]),
    ("King Lear", "King Lear", "tragedy", ["Lear", "Goneril", "Regan", "Cordelia", "Gloucester", "Edmund", "Edgar"]),
    ("Macbeth", "Macbeth", "tragedy", ["Macbeth", "Banquo", "Duncan", "Macduff", "Malcolm", "Fleance"]),
    ("Antony and Cleopatra", "Antony and Cleopatra", "tragedy", ["Antony", "Cleopatra", "Octavius", "Enobarbus", "Octavia", "Lepidus"]),
    ("Coriolanus", "Coriolanus", "tragedy", ["Coriolanus", "Volumnia", "Aufidius", "Menenius", "Virgilia", "Martius"]),
    ("Timon of Athens", "Timon of Athens", "tragedy (with Middleton)", ["Timon", "Apemantus", "Alcibiades", "Flavius"]),
    ("Sir Thomas More (play)", "Sir Thomas More", "collaborative; Shakespeare's hand in one scene", ["More", "Surrey", "Lincoln"]),
]


def main():
    rows = []
    for title, lab, sec, names in PLAYS:
        page = fetch_extract(title)
        lead, secs = split_sections(page.get("extract") or "")
        name, body = plot_section(page)
        s_all = sentences(body) if body else []
        # if the matching rule finds nothing, record the section names so we can see what the play uses
        rows.append(dict(title=page.get("title"), label=lab, section_kind=sec, names=names, plot_section=name,
                         n_sentences=len(s_all), n_sentences_capped=min(len(s_all), MAX_SENTENCES),
                         top_sections=[n for n, _ in secs][:12], sentences=s_all, raw_section=body))
        print(f"{lab:26s} {str(name):14s} {len(s_all):4d}  {'' if name else [n for n, _ in secs][:8]}")
    with open(SCOUT / "plays.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

# V8 Sexism Classification Prompt

Pass 2 of 2. Classifies whether a parliamentary speech contains hostile sexism, benevolent sexism, both, or neither, under Ambivalent Sexism Theory (Glick & Fiske, 1996). The stance label from Pass 1 is provided as context.

---

## Prompt

```
SYSTEM
You are classifying whether a parliamentary speech contains sexism toward women under Ambivalent Sexism Theory (Glick & Fiske, 1996, J. Pers. Soc. Psychol. 70(3): 491-512). Hostile and benevolent sexism are conceptualised as two independent dimensions of gendered prejudice; a single speech may contain either dimension, both, or neither. The speaker's stance on women's political rights is provided as context; your task here is only to classify sexism.

Hostile sexism degrades, blames, or seeks to control women. It frames women as incompetent, irrational, manipulative, or threatening, and reacts with resentment when women gain power or violate traditional roles. Following Glick & Fiske, three sub-types may apply:

- dominative_paternalism: women are incompetent and require male control or authority.
- competitive_gender_differentiation: men are more competent than women in the traits relevant to high-status domains.
- heterosexual_hostility: women's sexuality is framed as manipulative or threatening to men.

Benevolent sexism essentializes women -- it attributes admired traits to women as a class (purity, nurturance, morality, delicacy, special moral sensibility), typically expressed in chivalrous or protective terms, and uses that essentialization to justify restricting their roles, agency, or power. Following Glick & Fiske, three sub-types may apply:

- protective_paternalism: claims about the role of men toward women -- that men should care for, shield, decide for, sacrifice for, or rescue women.
- complementary_gender_differentiation: claims about the intrinsic qualities of women -- that women possess special purity, moral sensitivity, nurturance, or tenderness that men lack.
- heterosexual_intimacy: men are framed as incomplete without women as romantic partners.

The two dimensions are independent. A speech arguing "women are too emotional for politics but we must protect their special role" contains BOTH hostile (the incompetence claim) and benevolent sexism (protective idealisation).

Procedure. Decide the hostile and benevolent binary flags based on the definitions above, and list the subcategories that apply. If you mark hostile, you must list at least one hostile subcategory; if you mark benevolent, you must list at least one benevolent subcategory. Conversely, if no subcategory fits, do not mark the binary flag for that dimension.

Use the CONTEXT (preceding and following speech turns) to disambiguate references and to recognise framings the speaker is endorsing or responding to. Classify only the TARGET speech itself; sexist framings that appear only in the CONTEXT do not count toward the TARGET's classification.

If the TARGET text contains contributions from multiple speakers, mark hostile or benevolent if any substantive contribution exhibits the framing; the quote you cite must come from that contribution.

Each binary flag must be supported by a verbatim quote from the TARGET text.

USER
STANCE: {stance}

SPEECH:
{target_text}

CONTEXT:
{context_text}
```

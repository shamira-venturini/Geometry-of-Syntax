# Demo Prompt Generation Sanity Check

## Generation Summary

```csv
prompt_template,prime_condition,generation_class,n_items,total_items,share
demo__involving_event__did_to__mary_answered,active,agent_start_other,2,4,0.5
demo__involving_event__did_to__mary_answered,active,other,1,4,0.25
demo__involving_event__did_to__mary_answered,active,patient_start_other,1,4,0.25
demo__involving_event__did_to__mary_answered,filler,other,3,4,0.75
demo__involving_event__did_to__mary_answered,filler,patient_start_other,1,4,0.25
demo__involving_event__did_to__mary_answered,no_demo,agent_start_other,3,4,0.75
demo__involving_event__did_to__mary_answered,no_demo,patient_start_other,1,4,0.25
demo__involving_event__did_to__mary_answered,passive,agent_start_other,2,4,0.5
demo__involving_event__did_to__mary_answered,passive,patient_start_other,2,4,0.5
demo__involving_event__did_to__said_mary,active,agent_start_other,3,4,0.75
demo__involving_event__did_to__said_mary,active,patient_start_other,1,4,0.25
demo__involving_event__did_to__said_mary,filler,other,3,4,0.75
demo__involving_event__did_to__said_mary,filler,patient_start_other,1,4,0.25
demo__involving_event__did_to__said_mary,no_demo,agent_start_other,4,4,1.0
demo__involving_event__did_to__said_mary,passive,agent_start_other,2,4,0.5
demo__involving_event__did_to__said_mary,passive,patient_start_other,2,4,0.5
demo__involving_event__responsible_affected__mary_answered,active,agent_start_other,2,4,0.5
demo__involving_event__responsible_affected__mary_answered,active,other,1,4,0.25
demo__involving_event__responsible_affected__mary_answered,active,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__mary_answered,filler,other,3,4,0.75
demo__involving_event__responsible_affected__mary_answered,filler,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__mary_answered,no_demo,agent_start_other,3,4,0.75
demo__involving_event__responsible_affected__mary_answered,no_demo,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__mary_answered,passive,agent_start_other,2,4,0.5
demo__involving_event__responsible_affected__mary_answered,passive,other,1,4,0.25
demo__involving_event__responsible_affected__mary_answered,passive,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__said_mary,active,agent_start_other,2,4,0.5
demo__involving_event__responsible_affected__said_mary,active,other,1,4,0.25
demo__involving_event__responsible_affected__said_mary,active,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__said_mary,filler,other,3,4,0.75
demo__involving_event__responsible_affected__said_mary,filler,patient_start_other,1,4,0.25
demo__involving_event__responsible_affected__said_mary,no_demo,agent_start_other,4,4,1.0
demo__involving_event__responsible_affected__said_mary,passive,agent_start_other,2,4,0.5
demo__involving_event__responsible_affected__said_mary,passive,other,1,4,0.25
demo__involving_event__responsible_affected__said_mary,passive,patient_start_other,1,4,0.25
demo__there_was_event__did_to__mary_answered,active,agent_start_other,1,4,0.25
demo__there_was_event__did_to__mary_answered,active,other,3,4,0.75
demo__there_was_event__did_to__mary_answered,filler,other,4,4,1.0
demo__there_was_event__did_to__mary_answered,no_demo,agent_start_other,3,4,0.75
demo__there_was_event__did_to__mary_answered,no_demo,other,1,4,0.25
demo__there_was_event__did_to__mary_answered,passive,agent_start_other,2,4,0.5
demo__there_was_event__did_to__mary_answered,passive,other,1,4,0.25
demo__there_was_event__did_to__mary_answered,passive,patient_start_other,1,4,0.25
demo__there_was_event__did_to__said_mary,active,agent_start_other,1,4,0.25
demo__there_was_event__did_to__said_mary,active,other,3,4,0.75
demo__there_was_event__did_to__said_mary,filler,other,4,4,1.0
demo__there_was_event__did_to__said_mary,no_demo,agent_start_other,3,4,0.75
demo__there_was_event__did_to__said_mary,no_demo,other,1,4,0.25
demo__there_was_event__did_to__said_mary,passive,agent_start_other,2,4,0.5
demo__there_was_event__did_to__said_mary,passive,other,1,4,0.25
demo__there_was_event__did_to__said_mary,passive,patient_start_other,1,4,0.25
demo__there_was_event__responsible_affected__mary_answered,active,other,4,4,1.0
demo__there_was_event__responsible_affected__mary_answered,filler,other,4,4,1.0
demo__there_was_event__responsible_affected__mary_answered,no_demo,agent_start_other,3,4,0.75
demo__there_was_event__responsible_affected__mary_answered,no_demo,other,1,4,0.25
demo__there_was_event__responsible_affected__mary_answered,passive,other,3,4,0.75
demo__there_was_event__responsible_affected__mary_answered,passive,patient_start_other,1,4,0.25
demo__there_was_event__responsible_affected__said_mary,active,other,4,4,1.0
demo__there_was_event__responsible_affected__said_mary,filler,other,4,4,1.0
demo__there_was_event__responsible_affected__said_mary,no_demo,agent_start_other,4,4,1.0
demo__there_was_event__responsible_affected__said_mary,passive,other,4,4,1.0
```

## Generation Quality

```csv
prompt_template,prime_condition,total_items,extra_text_rate
demo__involving_event__did_to__mary_answered,active,4,1.0
demo__involving_event__did_to__mary_answered,filler,4,1.0
demo__involving_event__did_to__mary_answered,no_demo,4,0.75
demo__involving_event__did_to__mary_answered,passive,4,1.0
demo__involving_event__did_to__said_mary,active,4,1.0
demo__involving_event__did_to__said_mary,filler,4,1.0
demo__involving_event__did_to__said_mary,no_demo,4,0.5
demo__involving_event__did_to__said_mary,passive,4,1.0
demo__involving_event__responsible_affected__mary_answered,active,4,1.0
demo__involving_event__responsible_affected__mary_answered,filler,4,1.0
demo__involving_event__responsible_affected__mary_answered,no_demo,4,0.75
demo__involving_event__responsible_affected__mary_answered,passive,4,1.0
demo__involving_event__responsible_affected__said_mary,active,4,1.0
demo__involving_event__responsible_affected__said_mary,filler,4,1.0
demo__involving_event__responsible_affected__said_mary,no_demo,4,0.75
demo__involving_event__responsible_affected__said_mary,passive,4,1.0
demo__there_was_event__did_to__mary_answered,active,4,1.0
demo__there_was_event__did_to__mary_answered,filler,4,1.0
demo__there_was_event__did_to__mary_answered,no_demo,4,0.5
demo__there_was_event__did_to__mary_answered,passive,4,1.0
demo__there_was_event__did_to__said_mary,active,4,1.0
demo__there_was_event__did_to__said_mary,filler,4,1.0
demo__there_was_event__did_to__said_mary,no_demo,4,0.75
demo__there_was_event__did_to__said_mary,passive,4,1.0
demo__there_was_event__responsible_affected__mary_answered,active,4,1.0
demo__there_was_event__responsible_affected__mary_answered,filler,4,1.0
demo__there_was_event__responsible_affected__mary_answered,no_demo,4,1.0
demo__there_was_event__responsible_affected__mary_answered,passive,4,1.0
demo__there_was_event__responsible_affected__said_mary,active,4,1.0
demo__there_was_event__responsible_affected__said_mary,filler,4,1.0
demo__there_was_event__responsible_affected__said_mary,no_demo,4,0.75
demo__there_was_event__responsible_affected__said_mary,passive,4,1.0
```

## Examples

```csv
prompt_template,prime_condition,prompt,raw_generation,generated_first_sentence,generation_class
demo__there_was_event__responsible_affected__mary_answered,active,"A minister and a mother were involved in the same event.
There was a discovering event.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A minister discovers a mother.""

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," responsible person discovered a kid.""

A minister and a mother were involved in the same",The responsible person discovered a kid .,other
demo__there_was_event__responsible_affected__mary_answered,passive,"A minister and a mother were involved in the same event.
There was a discovering event.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A mother is discovered by a minister.""

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," responsible person was a husband.""

A husband and a kid were involved in the same",The responsible person was a husband .,other
demo__there_was_event__responsible_affected__mary_answered,no_demo,"A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband was drunk.""

Bridget asked, ""What happened?""

Mary",The husband was drunk .,agent_start_other
demo__there_was_event__responsible_affected__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The lantern glowed near sunset.""

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," lantern glowed near sunset.""

A husband and a kid were involved in the same",The lantern glowed near sunset .,other
demo__there_was_event__responsible_affected__said_mary,active,"A minister and a mother were involved in the same event.
There was a discovering event.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
""A minister discovers a mother."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," responsible person discovered a husband."" said Mary.

A husband and a kid were involved",The responsible person discovered a husband .,other
demo__there_was_event__responsible_affected__said_mary,passive,"A minister and a mother were involved in the same event.
There was a discovering event.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
""A mother is discovered by a minister."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," responsible person was a husband."" said Mary.

A husband and a kid were involved",The responsible person was a husband .,other
demo__there_was_event__responsible_affected__said_mary,no_demo,"A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," husband and the kid were involved in a car accident.

""The husband was driving",The husband and the kid were involved in a car accident .,agent_start_other
demo__there_was_event__responsible_affected__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The lantern glowed near sunset."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," lantern glowed near sunset."" said Mary.

A husband and a kid were involved",The lantern glowed near sunset .,other
demo__there_was_event__did_to__mary_answered,active,"A minister and a mother were involved in the same event.
There was a discovering event.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A minister discovers a mother.""

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," minister discovers a husband.""

A husband and a kid were involved in the same event",The minister discovers a husband .,other
demo__there_was_event__did_to__mary_answered,passive,"A minister and a mother were involved in the same event.
There was a discovering event.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A mother is discovered by a minister.""

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband was discovered by a minister.""

A husband and a kid were involved in the",The husband was discovered by a minister .,agent_start_other
demo__there_was_event__did_to__mary_answered,no_demo,"A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The", husband was in the car with the kid and the kid was in the car with the husband,The husband was in the car with the kid and the kid was in the car with the husband,agent_start_other
demo__there_was_event__did_to__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The lantern glowed near sunset.""

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," lantern glowed near sunset.""

A husband and a kid were involved in the same",The lantern glowed near sunset .,other
demo__there_was_event__did_to__said_mary,active,"A minister and a mother were involved in the same event.
There was a discovering event.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
""A minister discovers a mother."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," one who did it was a husband."" said Mary.

A man and a woman",The one who did it was a husband .,other
demo__there_was_event__did_to__said_mary,passive,"A minister and a mother were involved in the same event.
There was a discovering event.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
""A mother is discovered by a minister."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," husband and the kid were involved in the same event."" said Mary.

A man",The husband and the kid were involved in the same event .,agent_start_other
demo__there_was_event__did_to__said_mary,no_demo,"A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," husband and I were at a party and he was drunk,"" the husband said.

","The husband and I were at a party and he was drunk,",agent_start_other
demo__there_was_event__did_to__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The lantern glowed near sunset."" said Mary.

A husband and a kid were involved in the same event.
There was a striking event.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," lantern glowed near sunset."" said Mary.

A husband and a kid were involved",The lantern glowed near sunset .,other
demo__involving_event__responsible_affected__mary_answered,active,"There was a discovering event involving a minister and a mother.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A minister discovers a mother.""

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband discovered a kid.""

There was a striking event involving a husband and a wife",The husband discovered a kid .,agent_start_other
demo__involving_event__responsible_affected__mary_answered,passive,"There was a discovering event involving a minister and a mother.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A mother is discovered by a minister.""

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband was discovered by a minister.""

There was a striking event involving a husband and",The husband was discovered by a minister .,agent_start_other
demo__involving_event__responsible_affected__mary_answered,no_demo,"There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," kid got a little too close to the husband.""

Bridget asked, """,The kid got a little too close to the husband .,patient_start_other
demo__involving_event__responsible_affected__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The lantern glowed near sunset.""

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," kid was in the car.""

There was a striking event involving a husband and a",The kid was in the car .,patient_start_other
demo__involving_event__responsible_affected__said_mary,active,"There was a discovering event involving a minister and a mother.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
""A minister discovers a mother."" said Mary.

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," husband discovers a kid."" said Mary.

There was a striking event involving a husband",The husband discovers a kid .,agent_start_other
demo__involving_event__responsible_affected__said_mary,passive,"There was a discovering event involving a minister and a mother.
The responsible person was a minister.
The affected person was a mother.

Bridget asked, ""What happened?""
""A mother is discovered by a minister."" said Mary.

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," husband was discovered by a minister."" said Mary.

There was a striking event involving",The husband was discovered by a minister .,agent_start_other
demo__involving_event__responsible_affected__said_mary,no_demo,"There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The", husband was in the kitchen and he was cooking dinner. He was cooking a chicken and he,The husband was in the kitchen and he was cooking dinner .,agent_start_other
demo__involving_event__responsible_affected__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The lantern glowed near sunset."" said Mary.

There was a striking event involving a husband and a kid.
The responsible person was a husband.
The affected person was a kid.

Bridget asked, ""What happened?""
""The"," kid was in the car with the husband.""

""What happened?""

""",The kid was in the car with the husband .,patient_start_other
demo__involving_event__did_to__mary_answered,active,"There was a discovering event involving a minister and a mother.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A minister discovers a mother.""

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband discovered a kid.""

There was a discovery event involving a husband and a wife",The husband discovered a kid .,agent_start_other
demo__involving_event__did_to__mary_answered,passive,"There was a discovering event involving a minister and a mother.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
Mary answered, ""A mother is discovered by a minister.""

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," husband was discovered by a minister.""

There was a striking event involving a husband and",The husband was discovered by a minister .,agent_start_other
demo__involving_event__did_to__mary_answered,no_demo,"There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The", husband was in the kitchen and he heard a noise and he went to investigate. He found,The husband was in the kitchen and he heard a noise and he went to investigate .,agent_start_other
demo__involving_event__did_to__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The lantern glowed near sunset.""

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
Mary answered, ""The"," kid was in the car.""

There was a striking event involving a husband and a",The kid was in the car .,patient_start_other
demo__involving_event__did_to__said_mary,active,"There was a discovering event involving a minister and a mother.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
""A minister discovers a mother."" said Mary.

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," husband discovers a kid."" said Mary.

There was a striking event involving a husband",The husband discovers a kid .,agent_start_other
demo__involving_event__did_to__said_mary,passive,"There was a discovering event involving a minister and a mother.
The one who did it was a minister.
The one it happened to was a mother.

Bridget asked, ""What happened?""
""A mother is discovered by a minister."" said Mary.

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," husband is discovered by a minister."" said Mary.

There was a striking event involving",The husband is discovered by a minister .,agent_start_other
demo__involving_event__did_to__said_mary,no_demo,"There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," husband was in the kitchen and he was making a sandwich,"" said the husband.

","The husband was in the kitchen and he was making a sandwich,",agent_start_other
demo__involving_event__did_to__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The lantern glowed near sunset."" said Mary.

There was a striking event involving a husband and a kid.
The one who did it was a husband.
The one it happened to was a kid.

Bridget asked, ""What happened?""
""The"," kid was in the car with the husband.""

""What happened?""

""",The kid was in the car with the husband .,patient_start_other
demo__there_was_event__responsible_affected__mary_answered,active,"A bishop and a boy were involved in the same event.
There was a kissing event.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A bishop kisses a boy.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," bishop kissed a lawyer.""

A man and a woman were involved in the same event",The bishop kissed a lawyer .,other
demo__there_was_event__responsible_affected__mary_answered,passive,"A bishop and a boy were involved in the same event.
There was a kissing event.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A boy is kissed by a bishop.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the boy.""

A man and a woman were involved in the same event",The lawyer kissed the boy .,patient_start_other
demo__there_was_event__responsible_affected__mary_answered,no_demo,"A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lady was asking for a lawyer.""

Bridget asked, ""What happened?""",The lady was asking for a lawyer .,agent_start_other
demo__there_was_event__responsible_affected__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The traveler rested beside the river.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," traveler rested beside the river.""

A lady and a lawyer were involved in the same",The traveler rested beside the river .,other
demo__there_was_event__responsible_affected__said_mary,active,"A bishop and a boy were involved in the same event.
There was a kissing event.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
""A bishop kisses a boy."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," bishop kissed a boy."" said Mary.

A man and a woman were involved in",The bishop kissed a boy .,other
demo__there_was_event__responsible_affected__said_mary,passive,"A bishop and a boy were involved in the same event.
There was a kissing event.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
""A boy is kissed by a bishop."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," boy kissed the lady."" said Mary.

A man and a woman were involved in",The boy kissed the lady .,other
demo__there_was_event__responsible_affected__said_mary,no_demo,"A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," lady was asking for a lawyer.""

""What happened?""

""The lady",The lady was asking for a lawyer .,agent_start_other
demo__there_was_event__responsible_affected__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The traveler rested beside the river."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," traveler rested beside the river."" said Mary.

A lady and a lawyer were involved",The traveler rested beside the river .,other
demo__there_was_event__did_to__mary_answered,active,"A bishop and a boy were involved in the same event.
There was a kissing event.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A bishop kisses a boy.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," bishop kissed the lawyer.""

A man and a woman were involved in the same event",The bishop kissed the lawyer .,other
demo__there_was_event__did_to__mary_answered,passive,"A bishop and a boy were involved in the same event.
There was a kissing event.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A boy is kissed by a bishop.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the lady.""

A man and a woman were involved in the same event",The lawyer kissed the lady .,patient_start_other
demo__there_was_event__did_to__mary_answered,no_demo,"A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lady asked for a lawyer.""

Bridget asked, ""What did she want",The lady asked for a lawyer .,agent_start_other
demo__there_was_event__did_to__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The traveler rested beside the river.""

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," traveler rested beside the river.""

A lady and a lawyer were involved in the same",The traveler rested beside the river .,other
demo__there_was_event__did_to__said_mary,active,"A bishop and a boy were involved in the same event.
There was a kissing event.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
""A bishop kisses a boy."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," bishop kissed the lawyer."" said Mary.

A man and a woman were involved in",The bishop kissed the lawyer .,other
demo__there_was_event__did_to__said_mary,passive,"A bishop and a boy were involved in the same event.
There was a kissing event.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
""A boy is kissed by a bishop."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," lawyer kissed the lady."" said Mary.

A man and a woman were involved in",The lawyer kissed the lady .,patient_start_other
demo__there_was_event__did_to__said_mary,no_demo,"A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," lady asked me to do it.""

""What did you do?""

""",The lady asked me to do it .,agent_start_other
demo__there_was_event__did_to__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The traveler rested beside the river."" said Mary.

A lady and a lawyer were involved in the same event.
There was a asking event.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," traveler rested beside the river."" said Mary.

A lady and a lawyer were involved",The traveler rested beside the river .,other
demo__involving_event__responsible_affected__mary_answered,active,"There was a kissing event involving a bishop and a boy.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A bishop kisses a boy.""

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the lady.""

There was a asking event involving a man and a woman",The lawyer kissed the lady .,patient_start_other
demo__involving_event__responsible_affected__mary_answered,passive,"There was a kissing event involving a bishop and a boy.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A boy is kissed by a bishop.""

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the lady.""

There was a asking event involving a man and a woman",The lawyer kissed the lady .,patient_start_other
demo__involving_event__responsible_affected__mary_answered,no_demo,"There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lady was asking for a divorce.""

Bridget asked, ""What happened?""",The lady was asking for a divorce .,agent_start_other
demo__involving_event__responsible_affected__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The traveler rested beside the river.""

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," traveler rested beside the river.""

There was a asking event involving a lady and a",The traveler rested beside the river .,other
demo__involving_event__responsible_affected__said_mary,active,"There was a kissing event involving a bishop and a boy.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
""A bishop kisses a boy."" said Mary.

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," lawyer kissed the lady."" said Mary.

There was a asking event involving a man",The lawyer kissed the lady .,patient_start_other
demo__involving_event__responsible_affected__said_mary,passive,"There was a kissing event involving a bishop and a boy.
The responsible person was a bishop.
The affected person was a boy.

Bridget asked, ""What happened?""
""A boy is kissed by a bishop."" said Mary.

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," lawyer kissed the lady."" said Mary.

There was a asking event involving a man",The lawyer kissed the lady .,patient_start_other
demo__involving_event__responsible_affected__said_mary,no_demo,"There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," lady was asking for a lawyer,"" the lawyer said.

""What happened?""
","The lady was asking for a lawyer,",agent_start_other
demo__involving_event__responsible_affected__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The traveler rested beside the river."" said Mary.

There was a asking event involving a lady and a lawyer.
The responsible person was a lady.
The affected person was a lawyer.

Bridget asked, ""What happened?""
""The"," traveler rested beside the river."" said Mary.

There was a asking event involving a",The traveler rested beside the river .,other
demo__involving_event__did_to__mary_answered,active,"There was a kissing event involving a bishop and a boy.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A bishop kisses a boy.""

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the lady.""

There was a kissing event involving a bishop and a boy",The lawyer kissed the lady .,patient_start_other
demo__involving_event__did_to__mary_answered,passive,"There was a kissing event involving a bishop and a boy.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
Mary answered, ""A boy is kissed by a bishop.""

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer kissed the lady.""

There was a asking event involving a man and a woman",The lawyer kissed the lady .,patient_start_other
demo__involving_event__did_to__mary_answered,no_demo,"There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," lawyer said, 'I'm going to have to take you to court.'""

B","The lawyer said, 'I'm going to have to take you to court .",patient_start_other
demo__involving_event__did_to__mary_answered,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
Mary answered, ""The traveler rested beside the river.""

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
Mary answered, ""The"," traveler rested beside the river.""

There was a asking event involving a lady and a",The traveler rested beside the river .,other
demo__involving_event__did_to__said_mary,active,"There was a kissing event involving a bishop and a boy.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
""A bishop kisses a boy."" said Mary.

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," lawyer kissed the lady."" said Mary.

There was a kissing event involving a bishop",The lawyer kissed the lady .,patient_start_other
demo__involving_event__did_to__said_mary,passive,"There was a kissing event involving a bishop and a boy.
The one who did it was a bishop.
The one it happened to was a boy.

Bridget asked, ""What happened?""
""A boy is kissed by a bishop."" said Mary.

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," lawyer kissed the lady."" said Mary.

There was a asking event involving a man",The lawyer kissed the lady .,patient_start_other
demo__involving_event__did_to__said_mary,no_demo,"There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," lady said, 'I'm going to sue you for $10,000.'""

","The lady said, 'I'm going to sue you for $10,000 .",agent_start_other
demo__involving_event__did_to__said_mary,filler,"In another scene, something unrelated happened.
Bridget asked, ""What happened?""
""The traveler rested beside the river."" said Mary.

There was a asking event involving a lady and a lawyer.
The one who did it was a lady.
The one it happened to was a lawyer.

Bridget asked, ""What happened?""
""The"," traveler rested beside the river."" said Mary.

There was a asking event involving a",The traveler rested beside the river .,other
```
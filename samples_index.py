import os


cap_base = 'samples/caption'
cap = [
    dict(cap_objaid=os.path.splitext(x)[0], dispi=os.path.join(cap_base, x))
    for x in sorted(os.listdir(cap_base))
]

cls_base = 'samples/classification'
classification = [
    dict(cls_objaid=os.path.splitext(x)[0], dispi=os.path.join(cls_base, x))
    for x in sorted(os.listdir(cls_base))
]

sd_base = 'samples/sd'
sd_texts = {
    'b8db8dc5caad4fa5842a9ed6dbd2e9d6': 'falcon',
    'ff2875fb1a5b4771805a5fd35c8fe7bb': 'in the woods',
    'tpvzmLUXAURQ7ZxccJIBZvcIDlr': 'above the fields'
}
sd = [
    dict(
        sd_objaid=os.path.splitext(x)[0],
        dispi=os.path.join(sd_base, x),
        sdtprompt=sd_texts.get(os.path.splitext(x)[0], '')
    )
    for x in sorted(os.listdir(sd_base))
]

retrieval_texts = """
shark
swordfish
dolphin
goldfish
high heels
boots
slippers
sneakers
tiki mug
viking mug
animal-shaped mug
travel mug
white conical mug
green cubic mug
blue spherical mug
orange cylinder mug
""".splitlines()
retrieval_texts = [x.strip() for x in retrieval_texts if x.strip()]

pret_base = 'samples/retrieval-pc'
pret = [
    dict(retpc_objaid=os.path.splitext(x)[0], dispi=os.path.join(pret_base, x))
    for x in sorted(os.listdir(pret_base))
]

iret_base = 'samples/retrieval-img'
iret = [
    dict(rimageinput=os.path.join(iret_base, x), dispi=os.path.join(iret_base, x))
    for x in sorted(os.listdir(iret_base))
]

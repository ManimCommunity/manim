Procédure AfficherDec(L : ^cellule)

Var
    p, q : ^cellule

Début

    p ← L
    L ← Nil

    TantQue p <> Nil Faire

        q ← p
        p ← p^.suiv

        q^.suiv ← L
        L ← q

    FinTantQue

    p ← L

    TantQue p <> Nil Faire

        Ecrire(p^.val)
        p ← p^.suiv

    FinTantQue

Fin

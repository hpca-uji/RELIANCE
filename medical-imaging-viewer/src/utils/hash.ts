import bcrypt from 'bcrypt';

export async function hashPassword(password: string, saltRounds = 11): Promise<string> {
    const salt = bcrypt.genSaltSync(saltRounds);
    return bcrypt.hash(password, salt);
}

export async function validatePassword(plainPassword: string, hash: string): Promise<boolean> {
    return bcrypt.compare(plainPassword, hash);
}

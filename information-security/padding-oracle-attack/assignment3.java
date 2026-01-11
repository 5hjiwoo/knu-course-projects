import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class p_S2022428418 {

    static final int BLOCK_SIZE = 8;

    static byte[] hexToBytes(String s) {
        String n = s.trim();
        if (n.startsWith("0x")) n = n.substring(2);
        if ((n.length() % 2) == 1) n = "0" + n;
        byte[] out = new byte[n.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(n.substring(2 * i, 2 * i + 2), 16);
        }
        return out;
    }

    static String bytesToHex(byte[] b) {
        StringBuilder sb = new StringBuilder();
        for (byte x : b) sb.append(String.format("%02x", x & 0xff));
        return sb.toString();
    }

    static byte[] removePadding(byte[] block) throws Exception {
        if (block.length == 0) return block;
        int pad = block[block.length - 1] & 0xff;
        if (pad <= 0 || pad > block.length) throw new Exception("Invalid padding");
        for (int i = block.length - pad; i < block.length; i++) {
            if ((block[i] & 0xff) != pad) throw new Exception("Invalid padding");
        }
        return Arrays.copyOf(block, block.length - pad);
    }

    static byte[] recoverBlock(byte[] C0, byte[] C1, Oracle oracle) {
        final int B = C0.length;

        byte[] recovered = new byte[B];
        Arrays.fill(recovered, (byte) 0);

        java.util.function.BiFunction<Integer, byte[], List<Integer>> list = (idx, cur) -> {
            int i = idx;
            int p = B - i;
            List<Integer> ok = new ArrayList<>();
            for (int g = 0; g < 256; g++) {
                byte[] C0p = Arrays.copyOf(C0, B);
                for (int j = B - 1; j > i; j--) {
                    C0p[j] = (byte) ((C0[j] ^ cur[j]) ^ p);
                }
                C0p[i] = (byte) g;
                boolean res = oracle.check("0x" + bytesToHex(C0p), "0x" + bytesToHex(C1));
                if (res) ok.add(g);
            }
            return ok;
        };

        class DFS {
            boolean dfs(int i, byte[] cur) {
                if (i < 0) {
                    System.arraycopy(cur, 0, recovered, 0, B);
                    return true;
                }
                List<Integer> find = list.apply(i, cur);
                if (find.isEmpty()) return false;
                int p = B - i;
                for (int n : find) {
                    byte[] next = Arrays.copyOf(cur, B);
                    int m = (n ^ p) & 0xff;
                    next[i] = (byte) (m ^ (C0[i] & 0xff));
                    if (dfs(i - 1, next)) return true;
                }
                return false;
            }
        }

        boolean ok = new DFS().dfs(B - 1, new byte[B]);
        return recovered;
    }

    public static void main(String[] args) throws Exception {

        byte[] C0 = hexToBytes(args[0]);
        byte[] C1 = hexToBytes(args[1]);

        Oracle oracle = new Oracle();
        byte[] P1 = recoverBlock(C0, C1, oracle);

        byte[] plain;
        try {
            plain = removePadding(P1);
        } catch (Exception e) {
            plain = P1;
        }
        System.out.println(new String(plain, StandardCharsets.US_ASCII));
    }

    static class Oracle {
        private final pad_oracle oracleInstance;

        public Oracle() {
            this.oracleInstance = new pad_oracle();
        }

        public boolean check(String c0hex, String c1hex) {
            return oracleInstance.doOracle(c0hex, c1hex);
        }
    }
}
